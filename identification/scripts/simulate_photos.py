"""Fabrique de fausses photos de téléphone à partir des images de référence.

Perspective, fond (table, page d'album), éclairage, flou, compression JPEG,
oblitération. Sert à tester la robustesse en attendant de vraies photos ;
ne remplace pas une mesure sur de vrais clichés.

    python scripts/simulate_photos.py data/catalogue data/photos_simulees 3
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stampid.index import load_catalog, read_image


def background(h, w, rng):
    base = rng.integers(60, 230, 3)
    bg = np.full((h, w, 3), base, np.uint8).astype(np.int16)
    bg += rng.normal(0, 12, (h, w, 1)).astype(np.int16)  # grain du papier ou du bois
    return np.clip(bg, 0, 255).astype(np.uint8)


def cancel(img, rng):
    """Oblitération : quelques ondulations sombres sur une partie du timbre."""
    h, w = img.shape[:2]
    out = img.copy()
    y0 = rng.integers(0, h // 2)
    for k in range(rng.integers(3, 6)):
        xs = np.arange(0, w, 4)
        ys = (y0 + k * h // 12 + 6 * np.sin(xs / 15 + k)).astype(np.int32)
        cv2.polylines(out, [np.stack([xs, ys], 1)], False, (40, 40, 40), max(1, h // 150))
    return out


def simulate(ref, rng):
    h, w = ref.shape[:2]
    if rng.random() < 0.5:
        ref = cancel(ref, rng)
    H, W = int(max(h, w) * 2.2), int(max(h, w) * 1.8)
    canvas = background(H, W, rng)
    cx, cy = W / 2 + rng.normal(0, W * 0.05), H / 2 + rng.normal(0, H * 0.05)
    s = rng.uniform(0.8, 1.15)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) * s
    a = np.deg2rad(rng.uniform(-12, 12))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dst = dst @ rot.T + rng.normal(0, 0.04 * w, (4, 2)) + [cx, cy]
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(ref, M, (W, H))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    canvas[mask > 0] = warped[mask > 0]
    # Éclairage non uniforme, balance des blancs, flou, bruit, JPEG.
    yy, xx = np.mgrid[0:H, 0:W]
    light = 1 + 0.25 * ((xx / W - rng.random()) * rng.choice([-1, 1]))
    wb = rng.uniform(0.9, 1.1, 3)
    img = np.clip(canvas * light[..., None] * wb, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.6))
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    q = int(rng.integers(55, 90))
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def main(catalog_dir: str, out_dir: str, per_stamp: str = "3"):
    rng = np.random.default_rng(0)
    cat, out = Path(catalog_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = []
    for r in load_catalog(cat):
        ref = read_image(cat / r["image"])
        for k in range(int(per_stamp)):
            name = f"{r['id']}_{k}.jpg"
            cv2.imwrite(str(out / name), simulate(ref, rng))
            labels.append({"photo": name, "id": r["id"]})
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["photo", "id"])
        w.writeheader()
        w.writerows(labels)
    print(f"{len(labels)} photos simulées -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
