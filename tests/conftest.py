import struct
import tempfile
import wave
from pathlib import Path
import math

import pytest


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    """Generate a short WAV file with a 440Hz sine wave (1 second)."""
    filepath = tmp_path / "test_tone.wav"
    sample_rate = 22050
    duration = 1.0
    frequency = 440.0
    amplitude = 16000

    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        value = int(amplitude * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack("<h", value))

    with wave.open(str(filepath), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(samples))

    return filepath


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """Generate a short silent WAV file."""
    filepath = tmp_path / "silence.wav"
    sample_rate = 22050
    duration = 0.5
    n_samples = int(sample_rate * duration)

    with wave.open(str(filepath), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)

    return filepath
