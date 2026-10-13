"""Mesure le taux de bonne identification sur un dossier de photos.

Le dossier contient labels.csv (colonnes : photo, id) et les photos.
Affiche le top 1, le top 5 et les erreurs les plus fréquentes.

    python scripts/evaluate.py data/index.npz data/photos [--sans-verif] [--json res.json]
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stampid.embed_onnx import make_embedder
from stampid.index import StampIndex, identify, read_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("index")
    p.add_argument("photos")
    p.add_argument("--sans-verif", action="store_true", help="présélection seule, sans points clés")
    p.add_argument("--json", help="écrit les chiffres dans ce fichier")
    a = p.parse_args()
    index = StampIndex.load(Path(a.index))
    emb = make_embedder(index.model)
    photos = Path(a.photos)
    with open(photos / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    top1 = top5 = 0
    confusions = Counter()
    t = time.time()
    for r in rows:
        matches = identify(index, emb, read_image(photos / r["photo"]), k=5, rerank=not a.sans_verif)
        ids = [m.stamp_id for m in matches]
        top1 += ids[:1] == [r["id"]]
        top5 += r["id"] in ids
        if ids[:1] != [r["id"]]:
            confusions[(r["id"], ids[0] if ids else "-")] += 1
    n = len(rows)
    print(f"{n} photos, {len(index.ids)} timbres dans l'index, modèle {index.model}"
          f"{' sans vérification' if a.sans_verif else ''}")
    print(f"top 1 : {top1 / n:.1%}   top 5 : {top5 / n:.1%}   ({(time.time() - t) / n:.2f} s par photo)")
    for (truth, got), c in confusions.most_common(10):
        print(f"  {truth} pris pour {got} ({c} fois)")
    if a.json:
        Path(a.json).write_text(json.dumps({
            "modele": index.model, "verification": not a.sans_verif, "photos": n,
            "index": len(index.ids), "top1": top1 / n, "top5": top5 / n,
            "secondes_par_photo": (time.time() - t) / n,
            "confusions": [{"vrai": v, "trouve": g, "fois": c} for (v, g), c in confusions.most_common(20)],
        }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
