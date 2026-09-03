"""MODA SigLIP Distilled (HopitAI/moda-fashion-distilled) embedding endpoint.

Serves the 768-d, MIT-licensed embedding model used by Closet-theory-pipeline's
Stage 5 (app/providers/embedding/siglip.py).

Run with: flash dev
Deploy with: flash deploy
"""
import os

from runpod_flash import DataCenter, Endpoint, GpuGroup, NetworkVolume

moda_volume = NetworkVolume(name="moda-model-cache", size=20, datacenter=DataCenter.US_CA_2)


@Endpoint(
    name="moda-embed",
    gpu=GpuGroup.ADA_24,
    workers=(0, 2),
    idle_timeout=120,
    datacenter=DataCenter.US_CA_2,
    volume=moda_volume,
    env={
        "HF_HUB_CACHE": "/runpod-volume/hf-cache",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    },
    dependencies=[
        "torch",
        "open_clip_torch",
        "safetensors",
        "huggingface_hub",
        "pillow",
        "numpy",
    ],
)
class ModaEmbed:
    def __init__(self):
        import torch
        import open_clip
        from huggingface_hub import snapshot_download

        model_repo = "HopitAI/moda-fashion-distilled"
        model_dir = snapshot_download(repo_id=model_repo, cache_dir="/runpod-volume/hf-cache")
        weights = f"{model_dir}/open_clip_model.safetensors"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP",
            pretrained=weights,
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess

    async def embed(self, image_b64: str) -> dict:
        """Returns a normalized 768-d embedding for one image (base64-encoded bytes)."""
        import base64
        import io

        import torch
        import torch.nn.functional as F
        from PIL import Image

        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(tensor)
            emb = F.normalize(emb, p=2, dim=-1)

        return {"embedding": emb[0].float().cpu().tolist(), "dimension": emb.shape[-1]}
