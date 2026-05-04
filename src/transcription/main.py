import asyncio
import base64
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from transcription.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE, UPLOAD_DIR, OUTPUT_DIR
from transcription.transcriber import TranscribeParams, transcribe_audio

app = FastAPI(title="Audio to MIDI Transcription")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MIN_FILE_SIZE = 100

MAGIC_BYTES = {
    ".wav": [b"RIFF"],
    ".mp3": [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"ID3"],
    ".flac": [b"fLaC"],
    ".ogg": [b"OggS"],
}


def _validate_audio(content: bytes, ext: str) -> str | None:
    if len(content) < MIN_FILE_SIZE:
        return "ファイルが小さすぎます。有効な音声ファイルを選択してください。"
    expected = MAGIC_BYTES.get(ext, [])
    if expected:
        if not any(content[: len(m)] == m for m in expected):
            return f"ファイルの中身が{ext}形式ではありません。正しい音声ファイルを選択してください。"
    return None


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/score")
async def score_midi_endpoint(
    reference: UploadFile = File(...),
    predicted: UploadFile = File(...),
    onset_tolerance: float = Form(default=0.1),
    offset_tolerance: float = Form(default=0.2),
    pitch_tolerance: int = Form(default=0),
):
    from transcription.scoring import score_midi

    ref_content = await reference.read()
    pred_content = await predicted.read()
    ref_path = UPLOAD_DIR / f"ref_{uuid.uuid4().hex[:8]}.mid"
    pred_path = UPLOAD_DIR / f"pred_{uuid.uuid4().hex[:8]}.mid"
    try:
        ref_path.write_bytes(ref_content)
        pred_path.write_bytes(pred_content)
        result = score_midi(
            ref_path,
            pred_path,
            onset_tolerance=onset_tolerance,
            offset_tolerance=offset_tolerance,
            pitch_tolerance=pitch_tolerance,
        )
        return {
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "pitch_accuracy": result.pitch_accuracy,
            "onset_mae": result.onset_mae,
            "total_ref_notes": result.total_ref_notes,
            "total_pred_notes": result.total_pred_notes,
            "matched_notes": result.matched_notes,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"スコアリングに失敗しました: {e}")
    finally:
        ref_path.unlink(missing_ok=True)
        pred_path.unlink(missing_ok=True)


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
    separate_instruments: Optional[bool] = Form(default=None),
    preprocess: Optional[bool] = Form(default=None),
    drum_detail: Optional[bool] = Form(default=None),
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

    validation_err = _validate_audio(content, ext)
    if validation_err:
        raise HTTPException(status_code=400, detail=validation_err)

    upload_path = UPLOAD_DIR / file.filename
    upload_path.write_bytes(content)

    converted_path = None
    try:
        converted_path = _convert_if_needed(upload_path)
        actual_path = converted_path or upload_path

        params = _build_params(
            onset_threshold, frame_threshold, minimum_note_length_ms,
            min_velocity, quantize_enabled, quantize_strength,
            remove_pitch_bends, separate_instruments, preprocess, drum_detail,
        )
        result = transcribe_audio(actual_path, params)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"変換に失敗しました: {type(e).__name__}"
        )
    finally:
        upload_path.unlink(missing_ok=True)
        if converted_path:
            converted_path.unlink(missing_ok=True)

    if result.note_count == 0:
        raise HTTPException(
            status_code=422,
            detail="ノートが検出されませんでした。音声ファイルの内容を確認するか、感度設定を下げてみてください。",
        )

    headers = {
        "X-Detected-BPM": str(result.detected_bpm or ""),
        "X-Note-Count": str(result.note_count),
        "X-Stems-Used": ",".join(result.stems_used) if result.stems_used else "",
        "X-Time-Signature": result.time_signature or "",
        "X-Drum-Parts": ",".join(result.drum_parts) if result.drum_parts else "",
    }

    return FileResponse(
        path=str(result.midi_path),
        media_type="audio/midi",
        filename=f"{Path(file.filename).stem}.mid",
        headers=headers,
        background=_cleanup_task(result.midi_path),
    )


@app.post("/api/transcribe/batch")
async def transcribe_batch(
    files: list[UploadFile] = File(...),
    onset_threshold: Optional[float] = Form(default=None),
    frame_threshold: Optional[float] = Form(default=None),
    minimum_note_length_ms: Optional[float] = Form(default=None),
    min_velocity: Optional[int] = Form(default=None),
    quantize_enabled: Optional[bool] = Form(default=None),
    quantize_strength: Optional[float] = Form(default=None),
    remove_pitch_bends: Optional[bool] = Form(default=None),
    separate_instruments: Optional[bool] = Form(default=None),
    preprocess: Optional[bool] = Form(default=None),
    drum_detail: Optional[bool] = Form(default=None),
):
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="一度に変換できるのは最大20ファイルです。")

    params = _build_params(
        onset_threshold, frame_threshold, minimum_note_length_ms,
        min_velocity, quantize_enabled, quantize_strength,
        remove_pitch_bends, separate_instruments, preprocess, drum_detail,
    )

    results = []
    for file in files:
        if not file.filename:
            results.append({"filename": "unknown", "error": "No filename"})
            continue
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({"filename": file.filename, "error": f"非対応形式: {ext}"})
            continue
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            results.append({"filename": file.filename, "error": "ファイルサイズが大きすぎます"})
            continue
        validation_err = _validate_audio(content, ext)
        if validation_err:
            results.append({"filename": file.filename, "error": validation_err})
            continue

        upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
        upload_path.write_bytes(content)
        converted_path = None
        try:
            converted_path = _convert_if_needed(upload_path)
            actual_path = converted_path or upload_path
            r = transcribe_audio(actual_path, params)
            midi_bytes = r.midi_path.read_bytes()
            midi_b64 = base64.b64encode(midi_bytes).decode("ascii")
            results.append({
                "filename": file.filename,
                "midi_filename": f"{Path(file.filename).stem}.mid",
                "midi_base64": midi_b64,
                "detected_bpm": r.detected_bpm,
                "note_count": r.note_count,
                "stems_used": r.stems_used,
                "time_signature": r.time_signature,
                "drum_parts": r.drum_parts,
            })
            r.midi_path.unlink(missing_ok=True)
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
        finally:
            upload_path.unlink(missing_ok=True)
            if converted_path:
                converted_path.unlink(missing_ok=True)

    return {"results": results}


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    await websocket.accept()
    upload_path = None
    result_path = None
    converted_path = None

    try:
        try:
            data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
        except asyncio.TimeoutError:
            await websocket.send_json(
                {"type": "error", "detail": "接続がタイムアウトしました。"}
            )
            return

        filename = data.get("filename", "audio.wav")
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            await websocket.send_json(
                {"type": "error", "detail": f"Unsupported format: {ext}"}
            )
            return

        audio_bytes = base64.b64decode(data.get("audio_base64", ""))
        if len(audio_bytes) > MAX_UPLOAD_SIZE:
            await websocket.send_json(
                {"type": "error", "detail": "File too large. Maximum size is 50MB."}
            )
            return

        validation_err = _validate_audio(audio_bytes, ext)
        if validation_err:
            await websocket.send_json({"type": "error", "detail": validation_err})
            return

        upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{filename}"
        upload_path.write_bytes(audio_bytes)

        try:
            converted_path = _convert_if_needed(upload_path)
        except RuntimeError as e:
            await websocket.send_json({"type": "error", "detail": str(e)})
            return
        actual_path = converted_path or upload_path

        params = _build_params(
            data.get("onset_threshold"),
            data.get("frame_threshold"),
            data.get("minimum_note_length_ms"),
            data.get("min_velocity"),
            data.get("quantize_enabled"),
            data.get("quantize_strength"),
            data.get("remove_pitch_bends"),
            data.get("separate_instruments"),
            data.get("preprocess"),
            data.get("drum_detail"),
        )

        loop = asyncio.get_running_loop()
        disconnected = False

        async def _send_progress(step: str, pct: int):
            nonlocal disconnected
            if not disconnected:
                try:
                    await websocket.send_json(
                        {"type": "progress", "step": step, "percent": pct}
                    )
                except Exception:
                    disconnected = True

        def on_progress(step: str, pct: int):
            if not disconnected:
                asyncio.run_coroutine_threadsafe(_send_progress(step, pct), loop)

        result = await loop.run_in_executor(
            None, lambda: transcribe_audio(actual_path, params, on_progress)
        )
        result_path = result.midi_path

        if result.note_count == 0:
            await websocket.send_json(
                {
                    "type": "error",
                    "detail": "ノートが検出されませんでした。感度設定を下げてみてください。",
                }
            )
            return

        midi_bytes = result.midi_path.read_bytes()
        midi_b64 = base64.b64encode(midi_bytes).decode("ascii")

        await websocket.send_json(
            {
                "type": "result",
                "midi_base64": midi_b64,
                "filename": f"{Path(filename).stem}.mid",
                "detected_bpm": result.detected_bpm,
                "note_count": result.note_count,
                "stems_used": result.stems_used,
                "time_signature": result.time_signature,
                "drum_parts": result.drum_parts,
            }
        )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        if upload_path and upload_path.exists():
            upload_path.unlink(missing_ok=True)
        if result_path and result_path.exists():
            result_path.unlink(missing_ok=True)
        if converted_path and converted_path is not None:
            Path(converted_path).unlink(missing_ok=True)


def _convert_if_needed(upload_path: Path) -> Path | None:
    from transcription.convert import convert_to_wav

    return convert_to_wav(upload_path)


def _build_params(
    onset_threshold=None,
    frame_threshold=None,
    minimum_note_length_ms=None,
    min_velocity=None,
    quantize_enabled=None,
    quantize_strength=None,
    remove_pitch_bends=None,
    separate_instruments=None,
    preprocess=None,
    drum_detail=None,
) -> TranscribeParams:
    params = TranscribeParams()
    if onset_threshold is not None:
        params.onset_threshold = max(0.0, min(1.0, float(onset_threshold)))
    if frame_threshold is not None:
        params.frame_threshold = max(0.0, min(1.0, float(frame_threshold)))
    if minimum_note_length_ms is not None:
        params.minimum_note_length_ms = max(10.0, min(5000.0, float(minimum_note_length_ms)))
    if min_velocity is not None:
        params.min_velocity = max(0, min(127, int(min_velocity)))
    if quantize_enabled is not None:
        params.quantize_enabled = bool(quantize_enabled)
    if quantize_strength is not None:
        params.quantize_strength = max(0.0, min(1.0, float(quantize_strength)))
    if remove_pitch_bends is not None:
        params.remove_pitch_bends = bool(remove_pitch_bends)
    if separate_instruments is not None:
        params.separate_instruments = bool(separate_instruments)
    if preprocess is not None:
        params.preprocess = bool(preprocess)
    if drum_detail is not None:
        params.drum_detail = bool(drum_detail)
    return params


def _cleanup_task(path: Path):
    from starlette.background import BackgroundTask

    return BackgroundTask(path.unlink, missing_ok=True)
