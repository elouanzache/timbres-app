"""Transformer une image de timbre en vecteur comparable.

Deux signaux sont combinés :
- un descripteur profond (EfficientNetV2-S pré-entraîné ImageNet) pour le dessin ;
- un histogramme de couleur HSV, parce que deux timbres au même dessin
  peuvent ne différer que par la couleur, et donc par la cote.

Le modèle est interchangeable : un CLIP/DINOv2 pourra remplacer
EfficientNet dès que l'environnement pourra télécharger ses poids.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

COLOR_BINS = (12, 4, 4)  # teinte, saturation, valeur


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def color_histogram(img_bgr: np.ndarray) -> np.ndarray:
    """Histogramme HSV normalisé, en ignorant le fond blanc et les marges."""
    h, w = img_bgr.shape[:2]
    inner = img_bgr[h // 10 : h - h // 10, w // 10 : w - w // 10]
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > 25) | (hsv[..., 2] < 200)).astype(np.uint8)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, list(COLOR_BINS), [0, 180, 0, 256, 0, 256])
    return _l2(np.sqrt(hist.ravel()))


class Embedder:
    def __init__(self, input_size: int = 384):
        import keras

        self.input_size = input_size
        self.model = keras.applications.EfficientNetV2S(
            include_top=False, pooling="avg", weights="imagenet",
            input_shape=(input_size, input_size, 3),
        )

    def deep(self, imgs_bgr: list[np.ndarray]) -> np.ndarray:
        batch = np.stack(
            [cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2RGB), (self.input_size,) * 2) for i in imgs_bgr]
        ).astype(np.float32)
        return _l2(self.model.predict(batch, verbose=0))

    def embed(self, imgs_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        return self.deep(imgs_bgr), np.stack([color_histogram(i) for i in imgs_bgr])
