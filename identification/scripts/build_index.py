"""Construit l'index à partir d'un dossier catalogue (catalog.csv + images).

    python scripts/build_index.py data/catalogue data/index.npz [dinov2-small]

Le 3e argument choisit le modèle : efficientnet (défaut) ou un nom de
stampid/embed_onnx.MODELS téléchargé par scripts/fetch_model.py.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stampid.embed_onnx import make_embedder
from stampid.index import StampIndex


def main(catalog_dir: str, out: str, model: str = "efficientnet"):
    t = time.time()
    index = StampIndex.build(Path(catalog_dir), make_embedder(model), model=model)
    index.save(Path(out))
    print(f"{len(index.ids)} timbres indexés en {time.time() - t:.1f} s -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
