"""Construit l'index à partir d'un dossier catalogue (catalog.csv + images).

    python scripts/build_index.py data/catalogue data/index.npz
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stampid.embed import Embedder
from stampid.index import StampIndex


def main(catalog_dir: str, out: str):
    t = time.time()
    index = StampIndex.build(Path(catalog_dir), Embedder())
    index.save(Path(out))
    print(f"{len(index.ids)} timbres indexés en {time.time() - t:.1f} s -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
