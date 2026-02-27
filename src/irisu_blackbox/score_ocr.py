from __future__ import annotations

import re

import cv2
import numpy as np

from irisu_blackbox.config import Rect

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None


_DIGITS_RE = re.compile(r"\d+")


def extract_score(frame_bgr: np.ndarray, region: Rect, tesseract_cmd: str | None = None) -> int | None:
    if pytesseract is None:
        return None

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    crop = frame_bgr[region.top : region.bottom, region.left : region.right]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(
        bw,
        config="--psm 7 -c tessedit_char_whitelist=0123456789",
    )
    match = _DIGITS_RE.search(text)
    if not match:
        return None
    return int(match.group(0))
