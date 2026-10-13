"""Isoler le timbre dans une photo : trouver le quadrilatère et le redresser."""

import cv2
import numpy as np

OUT_SIZE = 384


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordonne 4 points : haut-gauche, haut-droit, bas-droit, bas-gauche."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def find_stamp_quad(img: np.ndarray) -> np.ndarray | None:
    """Retourne les 4 coins du plus grand contour rectangulaire, ou None.

    Les dents du timbre font un bord irrégulier : on ferme l'image binaire
    avant de chercher les contours, puis on prend l'enveloppe convexe.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    k = max(3, min(img.shape[:2]) // 80) | 1
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = img.shape[0] * img.shape[1]
    best = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        hull = cv2.convexHull(c)
        if cv2.contourArea(hull) < 0.08 * area_img:
            break
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.03 * peri, True)
        if len(approx) == 4:
            best = approx.reshape(4, 2).astype(np.float32)
            break
        if best is None:
            rect = cv2.minAreaRect(hull)
            best = cv2.boxPoints(rect).astype(np.float32)
    return None if best is None else _order_corners(best)


def crop_stamp(img: np.ndarray, size: int = OUT_SIZE) -> np.ndarray:
    """Redresse le timbre détecté ; sans détection, recadre au centre."""
    quad = find_stamp_quad(img)
    if quad is None:
        h, w = img.shape[:2]
        m = min(h, w)
        y0, x0 = (h - m) // 2, (w - m) // 2
        return cv2.resize(img[y0 : y0 + m, x0 : x0 + m], (size, size))
    tl, tr, br, bl = quad
    wq = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    hq = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    # On garde les proportions du timbre dans un carré de `size`.
    scale = size / max(wq, hq)
    ow, oh = int(round(wq * scale)), int(round(hq * scale))
    dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], dtype=np.float32)
    warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, dst), (ow, oh))
    canvas = np.full((size, size, 3), 255, np.uint8)
    y0, x0 = (size - oh) // 2, (size - ow) // 2
    canvas[y0 : y0 + oh, x0 : x0 + ow] = warped
    return canvas
