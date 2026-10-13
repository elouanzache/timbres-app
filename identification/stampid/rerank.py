"""Vérification géométrique : reclasser les candidats par points clés concordants.

Le descripteur global trouve des timbres « qui se ressemblent » ; SIFT +
RANSAC vérifie que c'est bien le même dessin, détail par détail. C'est
robuste à l'oblitération, à la perspective et à l'éclairage.
"""

import cv2
import numpy as np

_sift = cv2.SIFT_create(nfeatures=800)
_matcher = cv2.BFMatcher(cv2.NORM_L2)


def keypoints(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return _sift.detectAndCompute(gray, None)


def inliers(kp_a, kp_b) -> int:
    """Nombre de correspondances cohérentes avec une homographie."""
    (pa, da), (pb, db) = kp_a, kp_b
    if da is None or db is None or len(pa) < 8 or len(pb) < 8:
        return 0
    good = [m for m, n in (p for p in _matcher.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return len(good)
    src = np.float32([pa[m.queryIdx].pt for m in good])
    dst = np.float32([pb[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    return int(mask.sum()) if mask is not None else 0
