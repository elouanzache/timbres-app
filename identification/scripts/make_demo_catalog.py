"""Catalogue de démonstration : faux timbres générés, pour vérifier que la chaîne tourne.

Chaque dessin existe en plusieurs couleurs, comme les vrais timbres dont
seule la couleur change (et la cote avec). Ces images ne disent rien de la
précision sur de vrais timbres.

    python scripts/make_demo_catalog.py data/demo 40
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PALETTE = [(40, 40, 170), (150, 60, 30), (40, 120, 40), (120, 40, 120), (30, 110, 170)]


def perforate(img, rng):
    h, w = img.shape[:2]
    out = np.full((h + 24, w + 24, 3), 255, np.uint8)
    out[12 : 12 + h, 12 : 12 + w] = img
    step = 14
    for x in range(12, w + 12, step):
        for y in (6, h + 18):
            cv2.circle(out, (x, y), 4, (255, 255, 255), -1)
    return out


def design(seed: int, color, h=300, w=240):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(img, (14, 14), (w - 14, h - 14), color, 3)
    for _ in range(6):
        kind = rng.integers(3)
        c = tuple(int(v) for v in np.clip(np.array(color) + rng.integers(-30, 30, 3), 0, 255))
        if kind == 0:
            cv2.circle(img, tuple(int(v) for v in rng.integers(40, 200, 2)), int(rng.integers(15, 60)), c, -1)
        elif kind == 1:
            pts = rng.integers(30, 220, (5, 2)).astype(np.int32)
            cv2.fillPoly(img, [pts], c)
        else:
            p1, p2 = rng.integers(30, 220, (2, 2))
            cv2.line(img, tuple(int(v) for v in p1), tuple(int(v) for v in p2), c, int(rng.integers(3, 9)))
    cv2.putText(img, f"{int(rng.integers(1, 99))} c", (24, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    cv2.putText(img, "POSTES", (w - 130, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return perforate(img, rng)


def main(out_dir: str, n_designs: str = "40"):
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for d in range(int(n_designs)):
        for v, color in enumerate(PALETTE[: 1 + d % 3]):  # 1 à 3 variantes de couleur
            sid = f"demo{d:03d}_c{v}"
            cv2.imwrite(str(out / "images" / f"{sid}.png"), design(d, color))
            rows.append({"id": sid, "image": f"images/{sid}.png", "pays": "Demo", "titre": f"dessin {d}, couleur {v}"})
    with open(out / "catalog.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["id", "image", "pays", "titre"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} faux timbres -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
