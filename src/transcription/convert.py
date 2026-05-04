"""Audio format conversion — convert OGG/FLAC/M4A/etc. to WAV via ffmpeg."""

import shutil
import subprocess
import tempfile
from pathlib import Path

NEEDS_CONVERSION = {".ogg", ".m4a", ".aac", ".wma"}
NATIVE_FORMATS = {".wav", ".mp3", ".flac"}


def is_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def convert_to_wav(audio_path: Path) -> Path | None:
    ext = audio_path.suffix.lower()

    if ext in NATIVE_FORMATS:
        return None

    if ext not in NEEDS_CONVERSION:
        return None

    if not is_ffmpeg_available():
        raise RuntimeError(
            "このファイル形式の変換にはffmpegが必要です。\n"
            "インストール:\n"
            "  Mac: brew install ffmpeg\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )

    fd = tempfile.NamedTemporaryFile(suffix=".wav", prefix="converted_", delete=False)
    out_path = Path(fd.name)
    fd.close()

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ar", "22050", "-ac", "1", "-sample_fmt", "s16",
                str(out_path),
            ],
            capture_output=True,
            timeout=120,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpegでの変換に失敗しました: {e.stderr.decode(errors='replace')[:200]}"
        ) from e
    except FileNotFoundError:
        out_path.unlink(missing_ok=True)
        raise RuntimeError("ffmpegが見つかりません。インストールしてください。")

    return out_path
