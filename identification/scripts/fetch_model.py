"""Télécharge un modèle ONNX de Hugging Face dans data/models/<nom>.

    python scripts/fetch_model.py dinov2-small clip-b16
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stampid.embed_onnx import MODELS

ROOT = Path("data/models")


def fetch(name: str):
    repo, onnx_path = MODELS[name]
    dest = ROOT / name
    dest.mkdir(parents=True, exist_ok=True)
    for src, out in ((onnx_path, "model.onnx"), ("preprocessor_config.json", "preprocessor_config.json")):
        if not (dest / out).exists():
            url = f"https://huggingface.co/{repo}/resolve/main/{src}"
            print(f"{url} -> {dest / out}")
            urllib.request.urlretrieve(url, dest / out)


if __name__ == "__main__":
    for n in sys.argv[1:] or list(MODELS):
        fetch(n)
