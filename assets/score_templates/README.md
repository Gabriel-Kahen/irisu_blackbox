Place one template image per digit in this folder:

- `0.png`
- `1.png`
- `2.png`
- `3.png`
- `4.png`
- `5.png`
- `6.png`
- `7.png`
- `8.png`
- `9.png`

Notes:
- PNG/JPG/BMP/WEBP are accepted.
- Keep each image tightly cropped to a single score digit.
- Keep a little background around the glyph (1-3 px) so shape matching is stable.
- If templates are missing, score reading falls back to Tesseract when
  `env.score_ocr.template_fallback_to_tesseract = true`.
