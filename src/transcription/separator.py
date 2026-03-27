"""Audio source separation using Demucs (Meta).

Separates audio into stems: vocals, drums, bass, other.
Each stem can then be transcribed independently by Basic Pitch
for cleaner, instrument-specific MIDI tracks.
"""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


# MIDI program numbers for each stem type
STEM_PROGRAMS = {
    "vocals": 52,   # Choir Aahs
    "drums": 0,     # (will use is_drum=True)
    "bass": 33,     # Electric Bass (finger)
    "other": 0,     # Acoustic Grand Piano
}

STEM_NAMES = {
    "vocals": "Vocals",
    "drums": "Drums",
    "bass": "Bass",
    "other": "Piano/Other",
}


@dataclass
class SeparationResult:
    stems: dict[str, Path]  # stem name -> wav path
    work_dir: Path


def separate_audio(audio_path: Path) -> SeparationResult:
    """Separate audio into stems using Demucs htdemucs model.

    Args:
        audio_path: Path to the input audio file.

    Returns:
        SeparationResult with paths to separated stem WAV files.

    Raises:
        RuntimeError: If Demucs/torch is not installed or model cannot be loaded.
    """
    try:
        import torch
        import torchaudio
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError as e:
        raise RuntimeError(
            "楽器分離にはDemucsが必要です。"
            "インストール: pip install 'transcription[separation]'"
        ) from e

    work_dir = Path(tempfile.mkdtemp(prefix="demucs_"))

    # Load model (htdemucs = Hybrid Transformer, best quality)
    model = get_model("htdemucs")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load audio
    wav, sr = torchaudio.load(str(audio_path))

    # Resample to model's expected sample rate if needed
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        sr = model.samplerate

    # Ensure stereo (Demucs expects 2 channels)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    # Add batch dimension: (channels, samples) -> (1, channels, samples)
    wav = wav.unsqueeze(0).to(device)

    # Separate
    with torch.no_grad():
        sources = apply_model(model, wav, device=device)

    # sources shape: (1, n_sources, channels, samples)
    # model.sources: ['drums', 'bass', 'other', 'vocals']
    stems = {}
    for i, source_name in enumerate(model.sources):
        stem_audio = sources[0, i].cpu()  # (channels, samples)

        stem_path = work_dir / f"{source_name}.wav"
        torchaudio.save(str(stem_path), stem_audio, sr)
        stems[source_name] = stem_path

    return SeparationResult(stems=stems, work_dir=work_dir)


def is_separation_available() -> bool:
    """Check if Demucs and torch are installed."""
    try:
        import torch
        import torchaudio
        from demucs.pretrained import get_model
        return True
    except ImportError:
        return False


def cleanup_separation(result: SeparationResult) -> None:
    """Remove temporary stem files."""
    shutil.rmtree(result.work_dir, ignore_errors=True)
