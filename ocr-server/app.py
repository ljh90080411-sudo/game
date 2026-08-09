"""보스타임/창고 스크린샷에서 텍스트를 인식해주는 API 서버 (RapidOCR 기반, 경량).

POST /recognize_names  -> 빨강/파랑 닉네임만 색상 필터링 후 인식
POST /recognize_table  -> 필터 없이 표(PVP/창고) 텍스트 그대로 인식
"""

import base64
import io
import re

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from scipy.ndimage import binary_dilation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ocr = None


def get_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr import RapidOCR

        _ocr = RapidOCR(
            params={
                "Det.lang_type": "korean",
                "Rec.lang_type": "korean",
            }
        )
    return _ocr


class ImageIn(BaseModel):
    image: str  # base64 (data:image/...;base64,.... 형태도 허용)


def decode_image(image_b64: str) -> Image.Image:
    raw = image_b64.split(",")[-1]
    return Image.open(io.BytesIO(base64.b64decode(raw)))


def color_filter(img: Image.Image) -> Image.Image:
    """빨강/파랑 계열 글자만 검정으로 남기고 나머지는 흰 배경으로 지운다."""
    arr = np.array(img.convert("RGB")).astype(int)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    is_red = (r > 85) & ((r - g) > 22) & ((r - b) > 22) & (np.abs(g - b) < 40)
    is_blue = (b > 75) & ((b - r) > 22) & ((b - g) > 15) & (np.abs(r - g) < 45)
    mask = is_red | is_blue
    mask = binary_dilation(mask, iterations=1)

    out = np.where(mask[:, :, None], 0, 255).astype("uint8")
    out = np.repeat(out, 3, axis=2)
    return Image.fromarray(out)


def invert_filter(img: Image.Image) -> Image.Image:
    """어두운 배경 + 밝은 글자 화면을 OCR이 잘 읽는 형태로 단순 반전."""
    arr = np.array(img.convert("RGB")).astype(float)
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    val = (255 - lum).clip(0, 255).astype("uint8")
    out = np.repeat(val[:, :, None], 3, axis=2)
    return Image.fromarray(out)


JUNK_LINE = re.compile(
    r"(PM|AM|오전|오후)\s*\d{1,2}[:.]\d{2}|획득했습니다|경험치|나인을|드롭했습니다|입력하세요"
)


def extract_names(texts: list[str]) -> list[str]:
    names: list[str] = []
    for raw in texts:
        line = (raw or "").strip()
        if not line or JUNK_LINE.search(line):
            continue
        for tok in re.split(r"[,\s]+", line):
            tok = tok.strip()
            if 2 <= len(tok) <= 14 and not tok.isdigit() and tok not in names:
                names.append(tok)
    return names


def run_ocr(img: Image.Image) -> list[str]:
    ocr = get_ocr()
    result = ocr(np.array(img))
    if result is None or not result.txts:
        return []
    return list(result.txts)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/recognize_names")
def recognize_names(body: ImageIn):
    try:
        img = decode_image(body.image)
        filtered = color_filter(img)
        texts = run_ocr(filtered)
        return {"names": extract_names(texts)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@app.post("/recognize_table")
def recognize_table(body: ImageIn):
    try:
        img = decode_image(body.image)
        filtered = invert_filter(img)
        texts = run_ocr(filtered)
        return {"text": "\n".join(texts)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
