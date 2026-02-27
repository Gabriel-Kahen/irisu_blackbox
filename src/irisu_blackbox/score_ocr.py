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


def _ocr_digits_candidate(
    image: np.ndarray,
    *,
    tesseract_config: str,
) -> tuple[int | None, int, float]:
    data = pytesseract.image_to_data(
        image,
        config=tesseract_config,
        output_type=pytesseract.Output.DICT,
    )

    best_digits: str | None = None
    best_len = 0
    best_conf = -1.0

    for text, conf_raw in zip(data.get("text", []), data.get("conf", [])):
        digits = "".join(ch for ch in str(text) if ch.isdigit())
        if not digits:
            continue
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1.0
        if len(digits) > best_len or (len(digits) == best_len and conf > best_conf):
            best_digits = digits
            best_len = len(digits)
            best_conf = conf

    if best_digits is None:
        text = pytesseract.image_to_string(image, config=tesseract_config)
        matches = _DIGITS_RE.findall(text)
        if not matches:
            return None, 0, -1.0
        best_digits = max(matches, key=len)
        best_len = len(best_digits)
        best_conf = -1.0

    return int(best_digits), best_len, best_conf


def _build_score_variants(crop_bgr: np.ndarray) -> list[np.ndarray]:
    # Upscale first to improve OCR on thin stylized glyphs.
    scale = 3
    w = max(1, crop_bgr.shape[1] * scale)
    h = max(1, crop_bgr.shape[0] * scale)
    up_bgr = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(up_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hsv = cv2.cvtColor(up_bgr, cv2.COLOR_BGR2HSV)
    # Irisu score digits are yellow/gold with dark outlines; mask the warm range.
    yellow_mask = cv2.inRange(
        hsv,
        np.array([10, 35, 70], dtype=np.uint8),
        np.array([45, 255, 255], dtype=np.uint8),
    )
    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )

    return [
        otsu,
        cv2.bitwise_not(otsu),
        yellow_mask,
        cv2.bitwise_not(yellow_mask),
    ]


def extract_score(frame_bgr: np.ndarray, region: Rect, tesseract_cmd: str | None = None) -> int | None:
    if pytesseract is None:
        return None

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    crop = frame_bgr[region.top : region.bottom, region.left : region.right]
    if crop.size == 0:
        return None

    tesseract_config = (
        "--oem 1 --psm 7 "
        "-c classify_bln_numeric_mode=1 "
        "-c tessedit_char_whitelist=0123456789"
    )

    best_score: int | None = None
    best_len = 0
    best_conf = -1.0

    for variant in _build_score_variants(crop):
        score, digit_len, confidence = _ocr_digits_candidate(
            variant,
            tesseract_config=tesseract_config,
        )
        if score is None:
            continue
        if digit_len > best_len or (digit_len == best_len and confidence > best_conf):
            best_score = score
            best_len = digit_len
            best_conf = confidence

    if best_score is None:
        return None
    return best_score
