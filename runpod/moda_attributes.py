"""MODA_NER(V) attribute-extraction endpoint — crop / catalog / fullbody tracks.

Wraps the official inference suite from github.com/hopit-ai/Moda_ner (models/inference.py)
rather than re-implementing tensor loading. Serves Closet-theory-pipeline's Stage 3
(app/providers/attributes/moda_ner.py).

Run with: flash dev
Deploy with: flash deploy
"""
import os

from runpod_flash import DataCenter, Endpoint, GpuGroup, NetworkVolume

moda_volume = NetworkVolume(name="moda-model-cache", size=20, datacenter=DataCenter.US_CA_2)


@Endpoint(
    name="moda-ner",
    gpu=GpuGroup.ADA_24,
    workers=(0, 2),
    idle_timeout=120,
    datacenter=DataCenter.US_CA_2,
    volume=moda_volume,
    system_dependencies=["git"],
    env={
        "HF_HUB_CACHE": "/runpod-volume/hf-cache",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    },
    dependencies=[
        "torch",
        "open_clip_torch",
        "safetensors",
        "joblib",
        "numpy",
        "pillow",
        "huggingface_hub",
        "scikit-learn==1.5.2",
    ],
)
class ModaNer:
    def __init__(self):
        import subprocess
        import sys
        from pathlib import Path

        import torch

        self.track_repos = {
            "crop": "HopitAI/moda-ner-v-crop",
            "catalog": "HopitAI/moda-ner-v-catalog",
            "fullbody": "HopitAI/moda-ner-v-fullbody",
        }

        self.suite_dir = Path("/runpod-volume/Moda_ner")
        if not (self.suite_dir / "suite").exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/hopit-ai/Moda_ner", str(self.suite_dir)],
                check=True,
            )
        sys.path.insert(0, str(self.suite_dir))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._backends = {}  # lazily built per track, reused across requests on this worker

    def _get_backend(self, track: str):
        if track not in self.track_repos:
            raise ValueError(f"unknown track '{track}', expected one of {list(self.track_repos)}")
        if track not in self._backends:
            from huggingface_hub import snapshot_download
            from suite._model import ROUTES
            from suite._model.routes import LocalRoute

            model_dir = snapshot_download(
                repo_id=self.track_repos[track], cache_dir="/runpod-volume/hf-cache"
            )
            backend_cls = ROUTES[track]
            self._backends[track] = backend_cls(
                LocalRoute(model_dir, backend_cls.package_dirname), self.device
            )
        return self._backends[track]

    async def extract(self, image_b64: str, track: str) -> dict:
        """Runs one MODA_NER(V) track (crop | catalog | fullbody) on a single image."""
        import base64
        import tempfile
        from pathlib import Path

        backend = self._get_backend(track)

        image_bytes = base64.b64decode(image_b64)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            attributes = backend.predict(Path(tmp.name))

        return {"track": track, "attributes": attributes}
