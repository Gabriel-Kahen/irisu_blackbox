from pathlib import Path

import cv2
import numpy as np

from irisu_blackbox.config import Rect
from irisu_blackbox.score_ocr import extract_score_reading


def _draw_score_text(text: str) -> np.ndarray:
    slot_w = 26
    h = 34
    w = (len(text) * slot_w) + 8

    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:] = (25, 15, 70)  # dark red-ish background

    baseline = 26
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    thickness_outer = 5
    thickness_inner = 2

    for i, ch in enumerate(text):
        text_size = cv2.getTextSize(ch, font, font_scale, thickness_inner)[0]
        x = 4 + (i * slot_w) + ((slot_w - text_size[0]) // 2)
        cv2.putText(
            image,
            ch,
            (x, baseline),
            font,
            font_scale,
            (40, 10, 100),
            thickness_outer,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            ch,
            (x, baseline),
            font,
            font_scale,
            (120, 230, 245),
            thickness_inner,
            cv2.LINE_AA,
        )

    return image


def _tight_crop(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(mask)
    assert points is not None
    x, y, w, h = cv2.boundingRect(points)
    return image_bgr[y : y + h, x : x + w]


def _write_digit_templates(template_dir: Path) -> None:
    for digit in range(10):
        image = _draw_score_text(str(digit))
        crop = _tight_crop(image)
        ok = cv2.imwrite(str(template_dir / f"{digit}.png"), crop)
        assert ok


def test_template_reader_extracts_fixed_width_score(tmp_path: Path):
    _write_digit_templates(tmp_path)

    score_text = "00000298"
    score_image = _draw_score_text(score_text)

    frame = np.zeros((80, 260, 3), dtype=np.uint8)
    y0 = 20
    x0 = 12
    h, w = score_image.shape[:2]
    frame[y0 : y0 + h, x0 : x0 + w] = score_image

    reading = extract_score_reading(
        frame_bgr=frame,
        region=Rect(left=x0, top=y0, width=w, height=h),
        method="template",
        template_dir=str(tmp_path),
        template_fallback_to_tesseract=False,
    )

    assert reading is not None
    assert reading.score == 298
    assert reading.digit_len == len(score_text)
    assert reading.confidence > 50.0


def test_template_reader_returns_none_without_templates(tmp_path: Path):
    image = _draw_score_text("00000000")

    reading = extract_score_reading(
        frame_bgr=image,
        region=Rect(left=0, top=0, width=image.shape[1], height=image.shape[0]),
        method="template",
        template_dir=str(tmp_path),
        template_fallback_to_tesseract=False,
    )

    assert reading is None


def test_template_reader_accepts_prefixed_template_filenames(tmp_path: Path):
    for digit in range(10):
        image = _draw_score_text(str(digit))
        crop = _tight_crop(image)
        ok = cv2.imwrite(str(tmp_path / f"{digit}_copy.png"), crop)
        assert ok

    score_text = "00000298"
    score_image = _draw_score_text(score_text)

    reading = extract_score_reading(
        frame_bgr=score_image,
        region=Rect(left=0, top=0, width=score_image.shape[1], height=score_image.shape[0]),
        method="template",
        template_dir=str(tmp_path),
        template_fallback_to_tesseract=False,
    )

    assert reading is not None
    assert reading.score == 298
