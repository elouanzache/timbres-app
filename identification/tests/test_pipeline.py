"""Test de bout en bout sur quelques faux timbres : la chaîne doit retrouver le bon."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stampid.embed import Embedder
from stampid.index import StampIndex, identify, load_catalog, read_image


def test_retrouve_un_timbre_photographie(tmp_path):
    py = sys.executable
    subprocess.run([py, ROOT / "scripts/make_demo_catalog.py", tmp_path / "cat", "6"], check=True)
    subprocess.run([py, ROOT / "scripts/simulate_photos.py", tmp_path / "cat", tmp_path / "photos", "1"], check=True)
    emb = Embedder()
    index = StampIndex.build(tmp_path / "cat", emb)
    index.save(tmp_path / "index.npz")
    index = StampIndex.load(tmp_path / "index.npz")
    stamp = load_catalog(tmp_path / "cat")[0]["id"]
    matches = identify(index, emb, read_image(tmp_path / "photos" / f"{stamp}_0.jpg"))
    assert stamp in [m.stamp_id for m in matches]
