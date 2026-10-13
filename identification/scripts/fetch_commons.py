"""Télécharge des images de timbres depuis Wikimedia Commons pour un premier index.

Une catégorie par pays (ex. « Stamps of France », « Stamps of French Indochina »).
Garde la licence de chaque image dans catalog.csv. Les timbres anciens y sont
souvent dans le domaine public, mais chaque image doit être vérifiée avant
un usage commercial.

    python scripts/fetch_commons.py data/catalogue "Stamps of France=France" \
        "Stamps of French Indochina=Indochine" --max 300
"""

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "timbres-app-prototype/0.1 (identification de timbres)"}


def api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def category_files(category: str, limit: int, depth: int = 1):
    """Fichiers d'une catégorie et de ses sous-catégories (profondeur limitée)."""
    seen, queue, files = set(), [(category, 0)], []
    while queue and len(files) < limit:
        cat, d = queue.pop(0)
        if cat in seen:
            continue
        seen.add(cat)
        cont = {}
        while len(files) < limit:
            res = api({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}",
                       "cmtype": "file|subcat", "cmlimit": 500, **cont})
            for m in res["query"]["categorymembers"]:
                if m["ns"] == 6 and re.search(r"\.(jpe?g|png)$", m["title"], re.I):
                    files.append(m["title"])
                elif m["ns"] == 14 and d < depth:
                    queue.append((m["title"].removeprefix("Category:"), d + 1))
            if "continue" not in res:
                break
            cont = res["continue"]
    return files[:limit]


def image_info(titles: list[str]) -> dict:
    res = api({"action": "query", "titles": "|".join(titles), "prop": "imageinfo",
               "iiprop": "url|extmetadata", "iiurlwidth": 512})
    out = {}
    for p in res["query"]["pages"].values():
        if "imageinfo" in p:
            ii = p["imageinfo"][0]
            meta = ii.get("extmetadata", {})
            out[p["title"]] = {
                "url": ii.get("thumburl") or ii["url"],
                "page": ii.get("descriptionurl", ""),
                "licence": meta.get("LicenseShortName", {}).get("value", ""),
            }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("categories", nargs="+", help="'Catégorie Commons=Pays'")
    p.add_argument("--max", type=int, default=300, help="images par catégorie")
    a = p.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in a.categories:
        cat, _, pays = spec.partition("=")
        titles = category_files(cat, a.max)
        print(f"{cat} : {len(titles)} fichiers")
        for i in range(0, len(titles), 50):
            for title, info in image_info(titles[i : i + 50]).items():
                sid = re.sub(r"[^A-Za-z0-9]+", "_", title.removeprefix("File:").rsplit(".", 1)[0])[:80]
                dest = out / "images" / f"{sid}.jpg"
                if not dest.exists():
                    req = urllib.request.Request(info["url"], headers=UA)
                    dest.write_bytes(urllib.request.urlopen(req, timeout=30).read())
                    time.sleep(0.2)
                rows.append({"id": sid, "image": f"images/{sid}.jpg", "pays": pays or cat,
                             "titre": title.removeprefix("File:"), "licence": info["licence"],
                             "source": info["page"]})
    with open(out / "catalog.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["id", "image", "pays", "titre", "licence", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} timbres -> {out / 'catalog.csv'}")


if __name__ == "__main__":
    main()
