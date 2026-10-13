"""Identifie le timbre d'une photo et affiche les 5 candidats.

    python scripts/identify.py data/index.npz photo.jpg [--pays France]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stampid.embed import Embedder
from stampid.index import StampIndex, identify, read_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("index")
    p.add_argument("photo")
    p.add_argument("--pays", help="filtre sur la colonne 'pays' (ce que lira l'OCR)")
    a = p.parse_args()
    index = StampIndex.load(Path(a.index))
    where = {"pays": a.pays} if a.pays else None
    for rank, m in enumerate(identify(index, Embedder(), read_image(Path(a.photo)), where=where), 1):
        titre = m.meta.get("titre", "")
        print(f"{rank}. {m.stamp_id:<20} score {m.score:.3f}  {titre}")


if __name__ == "__main__":
    main()
