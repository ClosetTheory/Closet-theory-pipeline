"""RetinaFace-lineage face detector (SCRFD) + MobileSAM (garment segmentation) endpoint.

Serves Closet-theory-pipeline's Stage 2 (app/providers/detection/retina_sam.py).

Pipeline per request:
  1. SCRFD (insightface's "buffalo_l" detection model -- the maintained successor to the
     original RetinaFace, same author/lineage) finds the largest face. Run directly via
     bare onnxruntime rather than the `insightface` PyPI package: that package's
     __init__.py unconditionally imports its recognition/alignment code even when only
     detection is requested, dragging in scipy/scikit-image/scikit-learn/matplotlib/
     albumentations -- none of which detection needs -- and blows past Flash's 1.5GB
     build-artifact limit. The decode logic below (_ScrfdFaceDetector, nested inside
     __init__ since flash dev only ships decorated method bodies to the worker, not
     module-level helpers) is adapted verbatim from insightface/model_zoo/scrfd.py
     (MIT-licensed) so the math matches the reference implementation exactly.
  2. The face box seeds rough anatomical boxes for upper_body / lower_body (same
     projection heuristic as the OpenCV fallback provider) -- these are only *prompts*.
  3. Each seed box is fed to MobileSAM as a box prompt; SAM refines it into an actual
     foreground segmentation mask, and the garment region's bounding box is recomputed
     from the mask's true extent instead of the crude anatomical rectangle.
  4. If no face is found (flat-lay/catalog shot), a single full-frame box seeds SAM to
     find the dominant foreground object.

Run with: flash dev
Deploy with: flash deploy
"""
import os

from runpod_flash import DataCenter, Endpoint, GpuGroup, NetworkVolume

detect_volume = NetworkVolume(name="retina-sam-cache", size=15, datacenter=DataCenter.US_CA_2)


@Endpoint(
    name="retina-sam-detect",
    gpu=GpuGroup.ADA_24,
    workers=(0, 2),
    idle_timeout=120,
    datacenter=DataCenter.US_CA_2,
    volume=detect_volume,
    env={
        "HF_HUB_CACHE": "/runpod-volume/hf-cache",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    },
    dependencies=[
        # Bare onnxruntime (CPU) for the face detector -- deliberately NOT the
        # `insightface` package (drags in scipy/scikit-image/scikit-learn/matplotlib/
        # albumentations just from importing it) and NOT onnxruntime-gpu (bundles
        # CUDA/cuDNN, ~GBs). One face detection per request is fast enough on CPU.
        "onnxruntime",
        "opencv-python-headless",
        "numpy",
        "pillow",
        "torch",
        "timm",  # MobileSAM's TinyViT backbone; torchvision is not required.
        "huggingface_hub",
        "git+https://github.com/ChaoningZhang/MobileSAM.git",
    ],
)
class RetinaSamDetect:
    def __init__(self):
        import os
        import zipfile
        import urllib.request

        import torch
        from huggingface_hub import hf_hub_download
        from mobile_sam import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

        def _distance2bbox(points, distance):
            import numpy as np

            x1 = points[:, 0] - distance[:, 0]
            y1 = points[:, 1] - distance[:, 1]
            x2 = points[:, 0] + distance[:, 2]
            y2 = points[:, 1] + distance[:, 3]
            return np.stack([x1, y1, x2, y2], axis=-1)

        class _ScrfdFaceDetector:
            """SCRFD ONNX inference, adapted from insightface/model_zoo/scrfd.py
            (MIT license) to run standalone against a bare onnxruntime session --
            no `insightface` package, no `onnx` package, just onnxruntime + numpy +
            opencv."""

            def __init__(self, model_path: str, use_cuda: bool = False):
                import onnxruntime

                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
                )
                self.session = onnxruntime.InferenceSession(model_path, providers=providers)
                self.center_cache = {}
                self.nms_thresh = 0.4
                self.det_thresh = 0.5
                self.input_size = (640, 640)

                input_cfg = self.session.get_inputs()[0]
                self.input_name = input_cfg.name
                outputs = self.session.get_outputs()
                self.output_names = [o.name for o in outputs]

                self.input_mean = 127.5
                self.input_std = 128.0
                self.use_kps = False
                self._num_anchors = 1
                num_outputs = len(outputs)
                if num_outputs == 6:
                    self.fmc = 3
                    self._feat_stride_fpn = [8, 16, 32]
                    self._num_anchors = 2
                elif num_outputs == 9:
                    self.fmc = 3
                    self._feat_stride_fpn = [8, 16, 32]
                    self._num_anchors = 2
                    self.use_kps = True
                elif num_outputs == 10:
                    self.fmc = 5
                    self._feat_stride_fpn = [8, 16, 32, 64, 128]
                    self._num_anchors = 1
                elif num_outputs == 15:
                    self.fmc = 5
                    self._feat_stride_fpn = [8, 16, 32, 64, 128]
                    self._num_anchors = 1
                    self.use_kps = True
                else:
                    raise ValueError(f"Unexpected SCRFD output count: {num_outputs}")

            def prepare(self, det_thresh: float = 0.5, input_size=(640, 640)):
                self.det_thresh = det_thresh
                self.input_size = input_size

            def _forward(self, det_img, threshold):
                import cv2
                import numpy as np

                blob = cv2.dnn.blobFromImage(
                    det_img,
                    1.0 / self.input_std,
                    (det_img.shape[1], det_img.shape[0]),
                    (self.input_mean, self.input_mean, self.input_mean),
                    swapRB=True,
                )
                net_outs = self.session.run(self.output_names, {self.input_name: blob})

                input_height, input_width = blob.shape[2], blob.shape[3]
                fmc = self.fmc
                scores_list, bboxes_list = [], []
                for idx, stride in enumerate(self._feat_stride_fpn):
                    scores = net_outs[idx]
                    bbox_preds = net_outs[idx + fmc] * stride

                    height, width = input_height // stride, input_width // stride
                    key = (height, width, stride)
                    if key in self.center_cache:
                        anchor_centers = self.center_cache[key]
                    else:
                        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                        if self._num_anchors > 1:
                            anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape(
                                (-1, 2)
                            )
                        if len(self.center_cache) < 100:
                            self.center_cache[key] = anchor_centers

                    pos_inds = np.where(scores.ravel() >= threshold)[0]
                    bboxes = _distance2bbox(anchor_centers, bbox_preds)
                    # Keep scores 2D (K,1) here, like the reference -- raveling before
                    # indexing turns each stride's contribution into a 1D array, and
                    # vstack-ing 1D arrays of different lengths (0 candidates at one
                    # stride, 5 at another) fails since vstack treats them as rows that
                    # must have equal length. Ravel happens once, after the final vstack.
                    scores_list.append(scores[pos_inds])
                    bboxes_list.append(bboxes[pos_inds])
                return scores_list, bboxes_list

            def detect(self, img):
                """Returns (bboxes[N,5] as x1,y1,x2,y2,score, None). Resizes with
                aspect ratio preserved into self.input_size, letterboxed top-left,
                matching the reference."""
                import cv2
                import numpy as np

                im_ratio = float(img.shape[0]) / img.shape[1]
                model_ratio = float(self.input_size[1]) / self.input_size[0]
                if im_ratio > model_ratio:
                    new_height = self.input_size[1]
                    new_width = int(new_height / im_ratio)
                else:
                    new_width = self.input_size[0]
                    new_height = int(new_width * im_ratio)
                det_scale = float(new_height) / img.shape[0]
                resized_img = cv2.resize(img, (new_width, new_height))
                det_img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
                det_img[:new_height, :new_width, :] = resized_img

                scores_list, bboxes_list = self._forward(det_img, self.det_thresh)
                if sum(s.size for s in scores_list) == 0:
                    return np.empty((0, 5), dtype=np.float32), None

                scores = np.vstack(scores_list).ravel()
                order = scores.argsort()[::-1]
                bboxes = np.vstack(bboxes_list) / det_scale
                pre_det = np.hstack((bboxes, scores[:, None])).astype(np.float32, copy=False)
                pre_det = pre_det[order, :]

                keep = self._nms(pre_det)
                return pre_det[keep, :], None

            def _nms(self, dets):
                import numpy as np

                x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
                areas = (x2 - x1 + 1) * (y2 - y1 + 1)
                order = scores.argsort()[::-1]

                keep = []
                while order.size > 0:
                    i = order[0]
                    keep.append(i)
                    xx1 = np.maximum(x1[i], x1[order[1:]])
                    yy1 = np.maximum(y1[i], y1[order[1:]])
                    xx2 = np.minimum(x2[i], x2[order[1:]])
                    yy2 = np.minimum(y2[i], y2[order[1:]])

                    w = np.maximum(0.0, xx2 - xx1 + 1)
                    h = np.maximum(0.0, yy2 - yy1 + 1)
                    inter = w * h
                    ovr = inter / (areas[i] + areas[order[1:]] - inter)

                    inds = np.where(ovr <= self.nms_thresh)[0]
                    order = order[inds + 1]
                return keep

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Face detector weights: the same det_10g.onnx insightface's own downloader
        # fetches, pulled directly from insightface's official GitHub release so we
        # don't need the `insightface` package installed at all.
        model_dir = "/runpod-volume/scrfd-cache"
        os.makedirs(model_dir, exist_ok=True)
        det_path = f"{model_dir}/det_10g.onnx"
        if not os.path.exists(det_path):
            zip_path = f"{model_dir}/buffalo_l.zip"
            urllib.request.urlretrieve(
                "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
                zip_path,
            )
            with zipfile.ZipFile(zip_path) as zf:
                zf.extract("det_10g.onnx", model_dir)
            os.remove(zip_path)

        self.face_detector = _ScrfdFaceDetector(det_path, use_cuda=self.device == "cuda")
        self.face_detector.prepare(det_thresh=0.5, input_size=(640, 640))

        ckpt_path = hf_hub_download(repo_id="dhkim2810/MobileSAM", filename="mobile_sam.pt")
        sam = sam_model_registry["vit_t"](checkpoint=ckpt_path).to(self.device).eval()
        self.predictor = SamPredictor(sam)
        # Reserved for the no-face fallback path if box-prompted SAM ever needs it.
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    async def detect(self, image_b64: str) -> dict:
        import base64
        import io

        import cv2
        import numpy as np
        from PIL import Image

        image_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = pil_img.size
        np_img_rgb = np.array(pil_img)
        np_img_bgr = cv2.cvtColor(np_img_rgb, cv2.COLOR_RGB2BGR)

        bboxes, _ = self.face_detector.detect(np_img_bgr)
        person_detected = bboxes.shape[0] > 0
        face_box = None
        seed_regions = []

        if person_detected:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            largest = bboxes[int(np.argmax(areas))]
            fx1, fy1, fx2, fy2 = [float(v) for v in largest[:4]]
            face_box = [int(fx1), int(fy1), int(fx2), int(fy2)]
            fw, fh = fx2 - fx1, fy2 - fy1

            upper_seed = [
                max(0.0, fx1 - fw * 1.4),
                min(float(img_h - 1), fy1 + fh * 0.95),
                min(float(img_w), fx2 + fw * 1.0),
                min(float(img_h), fy1 + fh * 4.8),
            ]
            seed_regions.append(("upper_body", upper_seed))

            if upper_seed[3] < img_h - img_h * 0.15:
                lower_seed = [
                    max(0.0, fx1 - fw * 1.0),
                    upper_seed[3],
                    min(float(img_w), fx2 + fw * 1.0),
                    float(img_h),
                ]
                seed_regions.append(("lower_body", lower_seed))
        else:
            pad_x, pad_y = img_w * 0.05, img_h * 0.05
            seed_regions.append(("upper_body", [pad_x, pad_y, img_w - pad_x, img_h - pad_y]))

        self.predictor.set_image(np_img_rgb)

        garment_regions = []
        for label, seed_box in seed_regions:
            box_arr = np.array(seed_box, dtype=np.float32)
            masks, scores, _ = self.predictor.predict(box=box_arr, multimask_output=True)
            best_mask = masks[int(np.argmax(scores))]
            ys, xs = np.where(best_mask)
            if len(xs) == 0 or len(ys) == 0:
                box = [int(v) for v in seed_box]
            else:
                box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            garment_regions.append({"label": label, "box": box})

        return {
            "person_detected": person_detected,
            "face_box": face_box,
            "garment_regions": garment_regions,
        }
