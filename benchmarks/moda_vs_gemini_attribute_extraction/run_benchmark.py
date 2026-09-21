"""Benchmark MODA_NER (crop + fullbody tracks, structural-only, no OpenRouter top-up)
against Gemini (via OpenRouter) for garment attribute extraction, per the spec:
50-100 (here: 20, see report for the sample-size caveat) real wardrobe images, hard cases
included, ground truth hand-labeled, per-attribute accuracy / latency / cost / failure modes.
"""
import asyncio
import base64
import io
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(REPO_ROOT, ".env"))

import httpx
from app.config import settings
from app.providers.attributes.moda_ner import ModaNerAttributeExtractorProvider
from app.providers.attributes.moda_ner_mapping import TRACK_MAPPERS
from app.providers.attributes.gemini import GeminiAttributeExtractorProvider

# The source images are NOT committed to this repo (see README.md in this folder for why —
# provenance caution on 2 of the 10 ethnic-wear images). Point IMG_DIRS at wherever you've
# put a local copy of the 20-image test set to re-run this benchmark; ground_truth.json and
# results.json in this folder are the already-run, already-scored artifacts.
BENCH = os.environ.get("MODA_BENCH_IMAGE_DIR", os.path.join(REPO_ROOT, "benchmarks", "moda_vs_gemini_attribute_extraction", "images"))
IMG_DIRS = [os.path.join(BENCH, "bench_images"), BENCH, os.path.join(REPO_ROOT, "unknown_material_samples", "images")]

WESTERN_IDS = [
    "eb3324cb-50d5-4621-8a49-5c0890580fcb", "7e088fa0-2546-41cc-bc69-df529c7f2372",
    "225cb7d6-d10d-4015-ace4-f72b3502eff7", "2fbaafbb-e9d2-43e1-afe7-d7c37547a016",
    "e28cde22-5347-4852-9f12-bab734fbc4b3", "812a6592-9400-4e60-86ba-284447fed334",
    "382c210d-6f40-4b59-b0fa-63525d720b5a", "dcb9c6e9-e9fc-4812-a0b8-3e1d2118f81b",
    "6096f91f-0835-4f82-8a7e-c60bfe825942", "8c8cc402-f717-42b4-b402-2792c977d905",
]
ETHNIC_FILES = [
    "ethnic_01__saree.jpg", "ethnic_02__lehenga.jpg", "ethnic_03__kurta.jpg", "ethnic_04__kurta.jpg",
    "ethnic_05__dupatta.jpg", "ethnic_06__salwar.jpg", "ethnic_07__dupatta.jpg", "ethnic_08__dupatta.jpg",
    "ethnic_09__dupatta.jpg", "ethnic_10__dhoti.jpg",
]


def find_image(key):
    for d in IMG_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.startswith(key):
                return os.path.join(d, fn)
    return None


async def moda_structural(provider, image_bytes, track):
    t0 = time.time()
    try:
        raw = await provider._extract_structural(image_bytes, track)
        return {"ok": True, "attrs": raw, "latency_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "latency_s": round(time.time() - t0, 2)}


async def gemini_extract(provider, image_bytes):
    t0 = time.time()
    try:
        result = await provider.extract_attributes(image_bytes)
        return {"ok": True, "attrs": result.model_dump() if hasattr(result, "model_dump") else dict(result),
                "latency_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "latency_s": round(time.time() - t0, 2)}


async def main():
    moda = ModaNerAttributeExtractorProvider()
    gemini = GeminiAttributeExtractorProvider()

    keys = [(f, "ethnic") for f in ETHNIC_FILES] + [(k, "western") for k in WESTERN_IDS]
    results = []
    for i, (key, group) in enumerate(keys, 1):
        path = find_image(key)
        if not path:
            print(f"[{i}/{len(keys)}] MISSING: {key}")
            continue
        image_bytes = open(path, "rb").read()
        print(f"[{i}/{len(keys)}] {key} ({group}, {len(image_bytes)//1024}KB) ...", flush=True)

        crop = await moda_structural(moda, image_bytes, "crop")
        fullbody = await moda_structural(moda, image_bytes, "fullbody")
        gem = await gemini_extract(gemini, image_bytes)

        results.append({
            "key": key, "group": group, "file": os.path.basename(path),
            "moda_crop": crop, "moda_fullbody": fullbody, "gemini": gem,
        })
        print(f"    moda_crop={'OK' if crop['ok'] else 'FAIL:'+crop.get('error','')[:60]} "
              f"({crop['latency_s']}s)  moda_fullbody={'OK' if fullbody['ok'] else 'FAIL:'+fullbody.get('error','')[:60]} "
              f"({fullbody['latency_s']}s)  gemini={'OK' if gem['ok'] else 'FAIL:'+gem.get('error','')[:60]} ({gem['latency_s']}s)")

        out_path = os.path.join(BENCH, "results.json")
        json.dump(results, open(out_path, "w", encoding="utf-8"), indent=1, default=str)

    print(f"\nDone. {len(results)} images benchmarked. Results at {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
