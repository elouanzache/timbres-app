"""Télécharge des images de timbres depuis Wikimedia Commons pour un premier index.

Une catégorie par pays (ex. « Stamps of France », « Stamps of French Indochina »).
Garde la licence de chaque image dans catalog.csv. Les timbres anciens y sont
souvent dans le domaine public, mais chaque image doit être vérifiée avant
un usage commercial.

L'API de Commons répond souvent 429 (trop de requêtes) depuis une IP partagée :
le script lit alors les pages HTML des catégories et des fichiers, plus
lentement mais sans limite.

    python scripts/fetch_commons.py data/catalogue "Stamps of France=France" \
        "Stamps of French Indochina=Indochine" --max 300
    python scripts/fetch_commons.py data/catalogue   # reprend les images manquantes
"""

import argparse
import csv
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
WIKI = "https://commons.wikimedia.org/wiki/"
UA = {"User-Agent": "timbres-app-prototype/0.1 (https://github.com/elouanzache/timbres-app) python-urllib"}
THUMB_WIDTH = 500  # Commons ne sert que certaines largeurs (330, 500, 960...)

# Fichiers et sous-catégories qui ne sont pas des timbres seuls.
SKIP = re.compile(
    r"cover|envelop|letter|lettre|brief|postcard|carte|card|fdc|first.day|souvenir|sheet|feuille|"
    r"block|bloc|booklet|carnet|postmark|cancel|obliter|designer|engraver|people|museum|"
    r"exhibition|document|map|photograph|poster|cinderella|vignette|label|stationery|entier|"
    r"advert|seal|revenue|fiscal|telegram|pneumatique|coin|banknote|by_subject|by subject",
    re.I,
)


def get(url: str, tries: int = 6) -> bytes:
    """GET poli : respecte Retry-After sur 429 et ralentit à chaque échec."""
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            wait = float(e.headers.get("Retry-After") or 0)
            # Au-delà d'une minute d'attente demandée, on rend la main à l'appelant.
            if e.code not in (429, 500, 502, 503, 504) or k == tries - 1 or wait > 60:
                raise
            time.sleep(wait + 2 ** k)
    raise RuntimeError("inaccessible")


def api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    return json.loads(get(url, tries=2))


def category_files_api(category: str, limit: int, depth: int = 1):
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


def _page(title: str, query: str = "") -> str:
    return get(WIKI + urllib.parse.quote(title.replace(" ", "_")) + query).decode("utf-8")


def category_files_html(category: str, limit: int, depth: int = 3):
    """Même parcours que l'API, en lisant les pages de catégorie (largeur d'abord)."""
    seen, queue, files = set(), [(category, 0)], []
    while queue and len(files) < limit:
        cat, d = queue.pop(0)
        if cat in seen:
            continue
        seen.add(cat)
        query = ""
        while True:
            try:
                s = _page(f"Category:{cat}", query)
            except urllib.error.HTTPError:
                break
            sub = s.split('id="mw-subcategories"', 1)[1].split('id="mw-category-media"', 1)[0] \
                if 'id="mw-subcategories"' in s else ""
            for c in re.findall(r'<a href="/wiki/Category:[^"]+" title="Category:([^"]+)"', sub):
                c = html.unescape(c)
                if d < depth and not SKIP.search(c):
                    queue.append((c, d + 1))
            for f in re.findall(r'class="galleryfilename[^"]*" title="(File:[^"]+)"', s):
                f = html.unescape(f)
                if re.search(r"\.(jpe?g|png)$", f, re.I) and not SKIP.search(f) and f not in files:
                    files.append(f)
            nxt = re.search(r'href="(/w/index\.php\?title=Category:[^"]*filefrom=[^"]*)"[^>]*>next page', s)
            if not nxt or len(files) >= limit:
                break
            query = "?" + html.unescape(nxt.group(1)).split("?", 1)[1].split("&", 1)[1]
            time.sleep(0.5)
        time.sleep(0.5)
    return files[:limit]


def image_info_api(titles: list[str]) -> dict:
    res = api({"action": "query", "titles": "|".join(titles), "prop": "imageinfo",
               "iiprop": "url|extmetadata", "iiurlwidth": THUMB_WIDTH})
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


def image_info_html(title: str) -> dict | None:
    """URL de la vignette et licence, lues sur la page du fichier."""
    s = _page(title)
    m = re.search(r'href="(https://upload\.wikimedia\.org/wikipedia/commons/(\w)/(\w\w)/([^"?]+))[^"]*"', s)
    if not m:
        return None
    full, a, ab, name = m.groups()
    w = re.search(r'class="fileInfo">\(?\s*([\d,]+)\s*×', s)
    width = int(w.group(1).replace(",", "")) if w else 0
    ext = name.rsplit(".", 1)[-1].lower()
    url = full if width and width <= THUMB_WIDTH else \
        f"https://upload.wikimedia.org/wikipedia/commons/thumb/{a}/{ab}/{name}/{THUMB_WIDTH}px-{name}" + \
        (".png" if ext not in ("jpg", "jpeg", "png") else "")
    lic = re.findall(r'class="licensetpl_short[^"]*"[^>]*>([^<]*)<', html.unescape(s))
    return {"url": url, "page": WIKI + urllib.parse.quote(title.replace(" ", "_")), "licence": lic[0] if lic else ""}


def write_catalog(out: Path, rows: list[dict]):
    with open(out / "catalog.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["id", "image", "pays", "titre", "licence", "source", "url"])
        w.writeheader()
        w.writerows(rows)


def list_stamps(categories: list[str], limit: int, use_api: bool) -> list[dict]:
    """Étape 1 : la liste des timbres (titre, pays, licence, URL), sans les images."""
    rows, ids = [], set()
    for spec in categories:
        cat, _, pays = spec.partition("=")
        titles = None
        if use_api:
            try:
                titles = category_files_api(cat, limit)
            except urllib.error.HTTPError as e:
                print(f"API indisponible ({e.code}), lecture des pages HTML")
                use_api = False
        if titles is None:
            titles = category_files_html(cat, limit)
        print(f"{cat} : {len(titles)} fichiers", flush=True)
        infos = {}
        if use_api:
            for i in range(0, len(titles), 50):
                infos.update(image_info_api(titles[i : i + 50]))
        for title in titles:
            sid = re.sub(r"[^A-Za-z0-9]+", "_", title.removeprefix("File:").rsplit(".", 1)[0])[:80]
            if sid in ids:
                continue
            info = infos.get(title)
            if info is None:
                info = image_info_html(title)
                time.sleep(0.3)
            if not info:
                continue
            ids.add(sid)
            rows.append({"id": sid, "image": f"images/{sid}.jpg", "pays": pays or cat,
                         "titre": title.removeprefix("File:"), "licence": info["licence"],
                         "source": info["page"], "url": info["url"]})
    return rows


def download(out: Path, rows: list[dict]) -> int:
    """Étape 2 : les images manquantes. S'arrête proprement si Commons demande d'attendre."""
    todo = [r for r in rows if not (out / r["image"]).exists()]
    for k, r in enumerate(todo):
        try:
            data = get(r["url"], tries=3)
        except urllib.error.HTTPError as e:
            wait = float(e.headers.get("Retry-After") or 0)
            if e.code == 429:
                print(f"Commons limite les téléchargements (attendre {wait:.0f} s) : "
                      f"{len(todo) - k} images restantes, relancer la même commande plus tard")
                return len(todo) - k
            print(f"  ignoré {r['titre']} ({e.code})")
            continue
        (out / r["image"]).write_bytes(data)
        time.sleep(0.3)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("categories", nargs="*", help="'Catégorie Commons=Pays' (sans : reprend catalog.csv)")
    p.add_argument("--max", type=int, default=300, help="images par catégorie")
    p.add_argument("--html", action="store_true", help="ne pas essayer l'API")
    a = p.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    if a.categories:
        rows = list_stamps(a.categories, a.max, not a.html)
        write_catalog(out, rows)
        print(f"{len(rows)} timbres -> {out / 'catalog.csv'}", flush=True)
    else:
        with open(out / "catalog.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    left = download(out, rows)
    print(f"{len(rows) - left} images sur {len(rows)} dans {out / 'images'}")


if __name__ == "__main__":
    main()
