from pathlib import Path
import tempfile

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

ALLOWED_EXTENSIONS = {".mp3", ".wav"}

UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="transcription_upload_"))
OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="transcription_output_"))
