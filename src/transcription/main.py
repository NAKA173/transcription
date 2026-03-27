import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from transcription.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE, UPLOAD_DIR
from transcription.transcriber import TranscribeParams, transcribe_audio

app = FastAPI(title="Audio to MIDI Transcription")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile,
    onset_threshold: Optional[float] = Form(default=None),
    frame_threshold: Optional[float] = Form(default=None),
    minimum_note_length_ms: Optional[float] = Form(default=None),
    min_velocity: Optional[int] = Form(default=None),
    quantize_enabled: Optional[bool] = Form(default=None),
    quantize_strength: Optional[float] = Form(default=None),
    remove_pitch_bends: Optional[bool] = Form(default=None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB.")

    upload_path = UPLOAD_DIR / file.filename
    upload_path.write_bytes(content)

    params = TranscribeParams()
    if onset_threshold is not None:
        params.onset_threshold = max(0.0, min(1.0, onset_threshold))
    if frame_threshold is not None:
        params.frame_threshold = max(0.0, min(1.0, frame_threshold))
    if minimum_note_length_ms is not None:
        params.minimum_note_length_ms = max(50.0, min(1000.0, minimum_note_length_ms))
    if min_velocity is not None:
        params.min_velocity = max(0, min(127, min_velocity))
    if quantize_enabled is not None:
        params.quantize_enabled = quantize_enabled
    if quantize_strength is not None:
        params.quantize_strength = max(0.0, min(1.0, quantize_strength))
    if remove_pitch_bends is not None:
        params.remove_pitch_bends = remove_pitch_bends

    try:
        result = transcribe_audio(upload_path, params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        upload_path.unlink(missing_ok=True)

    # Return MIDI file with metadata in headers
    headers = {
        "X-Detected-BPM": str(result.detected_bpm or ""),
        "X-Note-Count": str(result.note_count),
    }

    return FileResponse(
        path=str(result.midi_path),
        media_type="audio/midi",
        filename=f"{Path(file.filename).stem}.mid",
        headers=headers,
        background=_cleanup_task(result.midi_path),
    )


def _cleanup_task(path: Path):
    """Return a background task that removes the file after response is sent."""
    from starlette.background import BackgroundTask

    return BackgroundTask(path.unlink, missing_ok=True)
