"""Index de référence : un vecteur par timbre du catalogue, recherche des plus proches."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .crop import crop_stamp
from .embed import Embedder
from .rerank import inliers, keypoints

COLOR_WEIGHT = 0.5  # part de la couleur dans la présélection
SHORTLIST = 20  # candidats vérifiés par points clés
RERANK_COLOR = 0.6  # départage les variantes de couleur d'un même dessin


@dataclass
class Match:
    stamp_id: str
    score: float
    meta: dict


def load_catalog(catalog_dir: Path) -> list[dict]:
    """catalog.csv : id, image, puis colonnes libres (pays, année, valeur faciale...)."""
    with open(catalog_dir / "catalog.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"image illisible : {path}")
    return img


class StampIndex:
    def __init__(self, ids, deep, color, meta, root: Path):
        self.ids, self.deep, self.color, self.meta, self.root = list(ids), deep, color, meta, Path(root)

    @classmethod
    def build(cls, catalog_dir: Path, embedder: Embedder, crop_refs: bool = True, batch: int = 32):
        rows = load_catalog(catalog_dir)
        deep, color = [], []
        for i in range(0, len(rows), batch):
            imgs = [read_image(catalog_dir / r["image"]) for r in rows[i : i + batch]]
            imgs = [crop_stamp(im) if crop_refs else _pad_square(im) for im in imgs]
            d, c = embedder.embed(imgs)
            deep.append(d)
            color.append(c)
        meta = {r["id"]: r for r in rows}
        return cls([r["id"] for r in rows], np.vstack(deep), np.vstack(color), meta, catalog_dir)

    def save(self, path: Path):
        np.savez_compressed(path, deep=self.deep, color=self.color, ids=np.array(self.ids))
        Path(str(path) + ".meta.json").write_text(
            json.dumps({"root": str(self.root.resolve()), "stamps": self.meta}, ensure_ascii=False)
        )

    @classmethod
    def load(cls, path: Path):
        z = np.load(path)
        meta = json.loads(Path(str(path) + ".meta.json").read_text())
        return cls(z["ids"].tolist(), z["deep"], z["color"], meta["stamps"], meta["root"])

    def search(self, deep_q: np.ndarray, color_q: np.ndarray, k: int = 5, where: dict | None = None):
        """Plus proches voisins. `where` filtre sur les métadonnées (ex. pays lu par OCR)."""
        score = (1 - COLOR_WEIGHT) * (self.deep @ deep_q) + COLOR_WEIGHT * (self.color @ color_q)
        if where:
            keep = np.array([all(self.meta[i].get(f) == v for f, v in where.items()) for i in self.ids])
            score = np.where(keep, score, -np.inf)
        top = np.argsort(-score)[:k]
        return [Match(self.ids[j], float(score[j]), self.meta[self.ids[j]]) for j in top if np.isfinite(score[j])]


def _pad_square(img: np.ndarray, size: int = 384) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    canvas = np.full((size, size, 3), 255, np.uint8)
    y0, x0 = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    canvas[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img
    return canvas


def identify(index: StampIndex, embedder: Embedder, photo_bgr: np.ndarray, k: int = 5, where=None,
             rerank: bool = True):
    """Présélection par vecteurs, puis vérification par points clés des meilleurs candidats."""
    crop = crop_stamp(photo_bgr)
    d, c = embedder.embed([crop])
    shortlist = index.search(d[0], c[0], k=SHORTLIST if rerank else k, where=where)
    if not rerank or not shortlist:
        return shortlist[:k]
    qk = keypoints(crop)
    inl = np.array([inliers(qk, keypoints(crop_stamp(read_image(index.root / m.meta["image"])))) for m in shortlist], float)
    color = np.array([index.color[index.ids.index(m.stamp_id)] @ c[0] for m in shortlist])
    final = inl / max(inl.max(), 1) + RERANK_COLOR * color
    order = np.argsort(-final)[:k]
    return [Match(shortlist[j].stamp_id, float(final[j]), {**shortlist[j].meta, "points_concordants": int(inl[j])})
            for j in order]
