# Timbres : scanner, identifier, estimer, raconter

App iPhone qui identifie un timbre à partir d'une photo, en estime le prix et raconte son histoire. Premier périmètre : timbres d'Europe et des colonies européennes. Modèle : gratuit pendant l'essai (5 timbres par jour), puis payant.

Cadrage complet : https://claude.ai/code/artifact/aa444a6b-c55a-43cd-81e3-2a6db4bb8b66

## Contenu

- `identification/` : prototype Python d'identification par image (ce dépôt aujourd'hui).
- `ios/` : app SwiftUI (à venir).
- `server/` : API, prix et récits (à venir).

## Prototype d'identification

La chaîne, pour une photo :

1. **Cadrer** (`stampid/crop.py`) : trouve le contour du timbre et le redresse.
2. **Décrire** (`stampid/embed.py`) : vecteur EfficientNetV2-S + histogramme de couleur.
3. **Présélectionner** (`stampid/index.py`) : les 20 timbres les plus proches dans l'index.
4. **Vérifier** (`stampid/rerank.py`) : points clés SIFT + RANSAC pour confirmer le dessin, la couleur départage les variantes.

```bash
cd identification
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Catalogue de démonstration (faux timbres) et fausses photos de téléphone
python scripts/make_demo_catalog.py data/demo 40
python scripts/simulate_photos.py data/demo data/photos 3
python scripts/build_index.py data/demo data/index.npz
python scripts/evaluate.py data/index.npz data/photos
python scripts/identify.py data/index.npz data/photos/demo004_c0_1.jpg

# Vrais timbres depuis Wikimedia Commons (accès réseau requis)
python scripts/fetch_commons.py data/commons "Stamps of France=France" "Stamps of French Indochina=Indochine" --max 300
# ou reprendre la liste déjà constituée (2 420 timbres, 16 pays et colonies)
mkdir -p data/commons && cp jeux/commons/catalog.csv data/commons/ && python scripts/fetch_commons.py data/commons

# DINOv2 ou CLIP (ONNX, Hugging Face) à la place d'EfficientNet
python scripts/fetch_model.py dinov2-small clip-b16
python scripts/build_index.py data/demo data/index_dino.npz dinov2-small
python scripts/evaluate.py data/index_dino.npz data/photos --sans-verif

pytest tests
```

Un catalogue est un dossier avec `catalog.csv` (colonnes `id`, `image`, puis libres : `pays`, `titre`, `licence`...) et les images. Une série de photos de test est un dossier avec `labels.csv` (`photo`, `id`).

### Résultats actuels

Sur 79 faux timbres générés (dont des variantes de couleur d'un même dessin) et 237 photos simulées (perspective, fond, éclairage, flou, oblitération) :

| Version | Top 1 | Top 5 |
| --- | --- | --- |
| Vecteur EfficientNet seul | 24 % | 47 % |
| + couleur | 53 % | 76 % |
| + vérification par points clés | 87 % | 96 % |

Ces chiffres valident la chaîne, pas la précision sur de vrais timbres. Prochaine mesure : un index de vrais timbres (Commons, puis Colnect) et 200 vraies photos prises au téléphone.

### Limites connues

- Les points clés des références sont recalculés à chaque requête (~0,9 s par photo sur 4 cœurs) : à mettre en cache pour un vrai catalogue.
- EfficientNet (ImageNet) est un point de départ ; un modèle type CLIP ou DINOv2 devrait mieux présélectionner.
- Pas encore de lecture du texte (pays, valeur faciale) pour filtrer : le paramètre `where` de `search` est prêt pour ça.
