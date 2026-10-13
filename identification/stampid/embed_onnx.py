"""Descripteurs DINOv2 ou CLIP au format ONNX, à la place d'EfficientNet.

Les modèles viennent des exports ONNX de Hugging Face (dépôts Xenova/...),
téléchargés par scripts/fetch_model.py. ONNX tourne sans PyTorch ni
TensorFlow et se convertit en Core ML pour l'iPhone.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .embed import _l2, color_histogram

MODELS = {
    "dinov2-small": ("Xenova/dinov2-small", "onnx/model.onnx"),
    "dinov2-base": ("Xenova/dinov2-base", "onnx/model.onnx"),
    "clip-b16": ("Xenova/clip-vit-base-patch16", "onnx/vision_model.onnx"),
}


class OnnxEmbedder:
    def __init__(self, model_dir: Path, input_size: int = 224):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "preprocessor_config.json").read_text())
        self.mean = np.array(cfg["image_mean"], np.float32)
        self.std = np.array(cfg["image_std"], np.float32)
        self.input_size = input_size
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        outs = [o.name for o in self.session.get_outputs()]
        # CLIP : image_embeds (projection) ; DINOv2 : pooler_output (jeton CLS normalisé).
        self.output = next((n for n in ("image_embeds", "pooler_output") if n in outs), outs[0])

    def _prep(self, img_bgr: np.ndarray) -> np.ndarray:
        # Le timbre entier, sans recadrage central : on étire au carré.
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (self.input_size,) * 2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255 - self.mean) / self.std
        return x.transpose(2, 0, 1)

    def deep(self, imgs_bgr: list[np.ndarray], batch: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(imgs_bgr), batch):
            x = np.stack([self._prep(im) for im in imgs_bgr[i : i + batch]])
            y = self.session.run([self.output], {self.input_name: x})[0]
            out.append(y[:, 0] if y.ndim == 3 else y)
        return _l2(np.vstack(out))

    def embed(self, imgs_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        return self.deep(imgs_bgr), np.stack([color_histogram(i) for i in imgs_bgr])


def make_embedder(name: str | None, models_dir: Path = Path("data/models")):
    """None ou 'efficientnet' : le modèle d'origine ; sinon un modèle ONNX de MODELS."""
    if not name or name == "efficientnet":
        from .embed import Embedder

        return Embedder()
    return OnnxEmbedder(Path(models_dir) / name)
