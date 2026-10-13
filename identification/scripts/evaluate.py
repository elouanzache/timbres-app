"""Mesure le taux de bonne identification sur un dossier de photos.

Le dossier contient labels.csv (colonnes : photo, id) et les photos.
Affiche le top 1, le top 5 et les erreurs les plus fréquentes.

    python scripts/evaluate.py data/index.npz data/photos
"""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stampid.embed import Embedder
from stampid.index import StampIndex, identify, read_image


def main(index_path: str, photos_dir: str):
    index = StampIndex.load(Path(index_path))
    emb = Embedder()
    photos = Path(photos_dir)
    with open(photos / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    top1 = top5 = 0
    confusions = Counter()
    for r in rows:
        matches = identify(index, emb, read_image(photos / r["photo"]), k=5)
        ids = [m.stamp_id for m in matches]
        top1 += ids[:1] == [r["id"]]
        top5 += r["id"] in ids
        if ids[:1] != [r["id"]]:
            confusions[(r["id"], ids[0] if ids else "-")] += 1
    n = len(rows)
    print(f"{n} photos, {len(index.ids)} timbres dans l'index")
    print(f"top 1 : {top1 / n:.1%}   top 5 : {top5 / n:.1%}")
    for (truth, got), c in confusions.most_common(10):
        print(f"  {truth} pris pour {got} ({c} fois)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
