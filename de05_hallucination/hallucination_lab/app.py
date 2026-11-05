from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .detectors import availability, detect
from .llm import DetectorFailure, DetectorUnavailable
from .schemas import DetectionResult, DetectRequest

app = FastAPI(title="Hallucination Lab", version="0.1.0", description="Đề 05 — kiểm chứng đầu ra LLM")
STATIC = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0", "models_loaded_on_demand": True}


@app.get("/methods")
def methods():
    return {"methods": availability()}


@app.post("/detect", response_model=DetectionResult)
def detect_claim(request: DetectRequest):
    try:
        return detect(request)
    except DetectorUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DetectorFailure as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/examples")
def examples():
    return {
        "examples": [
            {
                "title": "Sai số liệu",
                "claim": "Việt Nam gia nhập ASEAN năm 2000.",
                "context": "Việt Nam gia nhập ASEAN năm 1995.",
                "question": "Việt Nam gia nhập ASEAN năm nào?",
            },
            {
                "title": "Có căn cứ",
                "claim": "Hà Nội là thủ đô của Việt Nam.",
                "context": "Hà Nội là thủ đô của Việt Nam.",
                "question": "Thủ đô của Việt Nam là gì?",
            },
            {
                "title": "Thiếu bằng chứng",
                "claim": "Nghiên cứu này có 500 người tham gia.",
                "context": "Nghiên cứu đánh giá hiệu quả của phương pháp học mới. Báo cáo chưa nêu cỡ mẫu.",
                "question": "Nghiên cứu có bao nhiêu người tham gia?",
            },
            {
                "title": "Trích dẫn sai · dùng LLM",
                "claim": "Theo báo cáo A, độ chính xác đạt 99%.",
                "context": "Báo cáo A: độ chính xác đạt 71%. Báo cáo không ghi nhận kết quả 99%.",
                "question": "Báo cáo A ghi nhận độ chính xác bao nhiêu?",
            },
        ]
    }
