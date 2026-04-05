"""
Reference checks for mask refinement logic mirrored in C++ (foreground binary morphology + small blob cull).
"""

import cv2
import numpy as np


def refine_like_cpp(class_mask: np.ndarray, k: int = 5, iterations: int = 1, min_area: int = 80) -> np.ndarray:
    fg = (class_mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=iterations)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = np.zeros_like(fg)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(kept, [c], 0, 255, thickness=cv2.FILLED)
    out = class_mask.copy().astype(np.uint8)
    out[kept == 0] = 0
    return out


def test_refinement_reduces_noise():
    m = np.zeros((64, 64), dtype=np.uint8)
    m[20:40, 20:40] = 2
    m[5, 5] = 1
    r = refine_like_cpp(m, k=5, iterations=1, min_area=50)
    assert r[5, 5] == 0
    assert r[30, 30] == 2
