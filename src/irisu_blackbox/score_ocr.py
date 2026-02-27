from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np

from irisu_blackbox.config import Rect

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None


_DIGITS_RE = re.compile(r"\d+")
_TEMPLATE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
_TEMPLATE_CANVAS_SIZE = (36, 24)  # h, w
_TEMPLATE_MARGIN = 2
_TEMPLATE_CACHE: dict[str, dict[int, np.ndarray]] = {}


@dataclass(slots=True)
class ScoreOCRReading:
    score: int
    confidence: float
    digit_len: int


def _find_true_spans(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    if mask_1d.size == 0:
        return []
    padded = np.concatenate(([0], mask_1d.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _build_digit_mask(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
        hsv = None
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(otsu) > (0.6 * otsu.size):
        otsu = cv2.bitwise_not(otsu)

    if hsv is None:
        mask = otsu
    else:
        # Score glyphs are warm/yellow with a dark outline. Prefer warm mask when possible.
        warm_mask = cv2.inRange(
            hsv,
            np.array([8, 25, 45], dtype=np.uint8),
            np.array([55, 255, 255], dtype=np.uint8),
        )
        if np.count_nonzero(warm_mask) >= max(10, int(0.01 * warm_mask.size)):
            mask = warm_mask
        else:
            mask = otsu

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    return mask


def _normalize_digit_mask(mask: np.ndarray) -> np.ndarray:
    canvas_h, canvas_w = _TEMPLATE_CANVAS_SIZE
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    points = cv2.findNonZero(mask)
    if points is None:
        return canvas

    x, y, w, h = cv2.boundingRect(points)
    tight = mask[y : y + h, x : x + w]
    if tight.size == 0:
        return canvas

    max_w = max(1, canvas_w - (2 * _TEMPLATE_MARGIN))
    max_h = max(1, canvas_h - (2 * _TEMPLATE_MARGIN))
    scale = min(max_w / max(1, w), max_h / max(1, h))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))

    resized = cv2.resize(tight, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    ox = (canvas_w - out_w) // 2
    oy = (canvas_h - out_h) // 2
    canvas[oy : oy + out_h, ox : ox + out_w] = resized
    return canvas


def _segment_digit_masks(crop_bgr: np.ndarray) -> list[np.ndarray]:
    mask = _build_digit_mask(crop_bgr)
    if mask.size == 0:
        return []

    h, w = mask.shape[:2]
    col_counts = np.count_nonzero(mask, axis=0)
    min_col_pixels = max(2, int(h * 0.25))
    col_on = (col_counts >= min_col_pixels).astype(np.uint8)

    if col_on.size > 4:
        closed = cv2.morphologyEx(
            (col_on[None, :] * 255),
            cv2.MORPH_CLOSE,
            np.ones((1, 3), dtype=np.uint8),
        )
        col_on = (closed[0] > 0).astype(np.uint8)

    spans = _find_true_spans(col_on)
    if not spans:
        return []

    min_span = max(2, int(w * 0.02))
    max_span = max(8, int(w * 0.28))
    digit_masks: list[np.ndarray] = []

    for start, end in spans:
        span_w = end - start + 1
        if span_w < min_span or span_w > max_span:
            continue

        slab = mask[:, start : end + 1]
        row_counts = np.count_nonzero(slab, axis=1)
        on_rows = np.flatnonzero(row_counts > 0)
        if on_rows.size == 0:
            continue

        top = int(on_rows[0])
        bottom = int(on_rows[-1]) + 1
        if (bottom - top) < max(4, int(h * 0.3)):
            continue

        digit_masks.append(slab[top:bottom, :])

    # The game score is fixed-width in practice; keep the right-most plausible run.
    if len(digit_masks) > 8:
        digit_masks = digit_masks[-8:]
    return digit_masks


def _find_template_file(template_dir: Path, digit: int) -> Path | None:
    for suffix in _TEMPLATE_SUFFIXES:
        candidate = template_dir / f"{digit}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _load_digit_templates(template_dir: str | None) -> dict[int, np.ndarray]:
    if not template_dir:
        return {}

    base = Path(template_dir).expanduser()
    try:
        base = base.resolve()
    except Exception:
        pass

    key = str(base)
    cached = _TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached

    templates: dict[int, np.ndarray] = {}
    if not base.exists() or not base.is_dir():
        _TEMPLATE_CACHE[key] = templates
        return templates

    for digit in range(10):
        template_path = _find_template_file(base, digit)
        if template_path is None:
            continue

        image = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if image is None:
            image = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        mask = _build_digit_mask(image)
        norm = _normalize_digit_mask(mask)
        if np.count_nonzero(norm) == 0:
            continue
        templates[digit] = norm

    _TEMPLATE_CACHE[key] = templates
    return templates


def _binary_dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a_mask = a > 0
    b_mask = b > 0
    a_count = int(np.count_nonzero(a_mask))
    b_count = int(np.count_nonzero(b_mask))
    if a_count == 0 or b_count == 0:
        return 0.0

    inter = int(np.count_nonzero(a_mask & b_mask))
    return float((2.0 * inter) / (a_count + b_count))


def _extract_score_with_templates(
    frame_bgr: np.ndarray,
    region: Rect,
    *,
    template_dir: str | None,
    min_similarity: float,
) -> ScoreOCRReading | None:
    templates = _load_digit_templates(template_dir)
    if len(templates) < 10:
        return None

    crop = frame_bgr[region.top : region.bottom, region.left : region.right]
    if crop.size == 0:
        return None

    digit_masks = _segment_digit_masks(crop)
    if not digit_masks:
        return None

    digits: list[str] = []
    sims: list[float] = []

    for raw_digit in digit_masks:
        norm_digit = _normalize_digit_mask(raw_digit)
        best_digit: int | None = None
        best_sim = -1.0

        for digit, template in templates.items():
            sim = _binary_dice_score(norm_digit, template)
            if sim > best_sim:
                best_sim = sim
                best_digit = digit

        if best_digit is None:
            return None

        digits.append(str(best_digit))
        sims.append(best_sim)

    if not digits:
        return None

    if min_similarity > 0 and any(sim < min_similarity for sim in sims):
        return None

    text = "".join(digits)
    if not text.isdigit():
        return None

    score = int(text)
    confidence = float(np.mean(sims) * 100.0)
    return ScoreOCRReading(score=score, confidence=confidence, digit_len=len(text))


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


def _is_better_candidate(
    digit_len: int,
    confidence: float,
    best_len: int,
    best_conf: float,
) -> bool:
    if best_len == 0:
        return True

    conf_known = confidence >= 0
    best_conf_known = best_conf >= 0
    if conf_known and best_conf_known:
        if abs(confidence - best_conf) > 1e-6:
            return confidence > best_conf
        return digit_len > best_len
    if conf_known and not best_conf_known:
        return True
    if not conf_known and best_conf_known:
        return False
    return digit_len > best_len


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


def _extract_score_with_tesseract(
    frame_bgr: np.ndarray,
    region: Rect,
    *,
    tesseract_cmd: str | None,
) -> ScoreOCRReading | None:
    if pytesseract is None:
        return None

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    crop = frame_bgr[region.top : region.bottom, region.left : region.right]
    if crop.size == 0:
        return None

    best_score: int | None = None
    best_len = 0
    best_conf = -1.0

    base_cfg = "-c classify_bln_numeric_mode=1 -c tessedit_char_whitelist=0123456789"
    tesseract_configs = [
        f"--oem 1 --psm 7 {base_cfg}",
        f"--oem 1 --psm 8 {base_cfg}",
        f"--oem 1 --psm 13 {base_cfg}",
    ]

    for variant in _build_score_variants(crop):
        for tesseract_config in tesseract_configs:
            score, digit_len, confidence = _ocr_digits_candidate(
                variant,
                tesseract_config=tesseract_config,
            )
            if score is None:
                continue
            if _is_better_candidate(digit_len, confidence, best_len, best_conf):
                best_score = score
                best_len = digit_len
                best_conf = confidence

    if best_score is None:
        return None
    return ScoreOCRReading(score=best_score, confidence=best_conf, digit_len=best_len)


def extract_score_reading(
    frame_bgr: np.ndarray,
    region: Rect,
    *,
    method: str = "tesseract",
    tesseract_cmd: str | None = None,
    template_dir: str | None = None,
    template_min_similarity: float = 0.32,
    template_fallback_to_tesseract: bool = True,
) -> ScoreOCRReading | None:
    method_key = method.strip().lower()

    if method_key in {"template", "auto"}:
        template_reading = _extract_score_with_templates(
            frame_bgr=frame_bgr,
            region=region,
            template_dir=template_dir,
            min_similarity=max(0.0, float(template_min_similarity)),
        )
        if template_reading is not None:
            return template_reading
        if method_key == "template" and not template_fallback_to_tesseract:
            return None

    return _extract_score_with_tesseract(
        frame_bgr=frame_bgr,
        region=region,
        tesseract_cmd=tesseract_cmd,
    )


def extract_score(
    frame_bgr: np.ndarray,
    region: Rect,
    tesseract_cmd: str | None = None,
    *,
    method: str = "tesseract",
    template_dir: str | None = None,
    template_min_similarity: float = 0.32,
    template_fallback_to_tesseract: bool = True,
) -> int | None:
    reading = extract_score_reading(
        frame_bgr=frame_bgr,
        region=region,
        method=method,
        tesseract_cmd=tesseract_cmd,
        template_dir=template_dir,
        template_min_similarity=template_min_similarity,
        template_fallback_to_tesseract=template_fallback_to_tesseract,
    )
    if reading is None:
        return None
    return reading.score
