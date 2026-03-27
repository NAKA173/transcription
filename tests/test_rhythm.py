from pathlib import Path

import numpy as np

from transcription.rhythm import BeatInfo, detect_beats, quantize_note_times


def test_detect_beats_returns_info(sample_wav: Path):
    """Beat detection should return a BeatInfo (tempo may be 0 for very short clips)."""
    info = detect_beats(sample_wav)
    assert info.tempo >= 0
    assert info.beat_interval > 0


def test_subdivisions_are_finer_than_beats():
    """Subdivisions should have ~4x more points than beat times."""
    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    info = BeatInfo(tempo=120.0, beat_times=beats, beat_interval=0.5)
    subs = info.subdivisions
    # 4 beats * 4 subdivisions + 1 final beat = 17
    assert len(subs) == 17


def test_quantize_snaps_to_grid():
    """Notes should snap toward nearest grid point."""
    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    # Note at 0.12 should snap toward 0.0
    q_start, q_end = quantize_note_times(0.12, 0.42, grid, strength=1.0)
    assert abs(q_start - 0.0) < 0.01

    # Note at 0.48 should snap toward 0.5
    q_start, q_end = quantize_note_times(0.48, 0.78, grid, strength=1.0)
    assert abs(q_start - 0.5) < 0.01


def test_quantize_preserves_duration():
    """Quantization should preserve note duration."""
    grid = np.array([0.0, 0.5, 1.0])
    duration = 0.3
    q_start, q_end = quantize_note_times(0.12, 0.12 + duration, grid, strength=1.0)
    assert abs((q_end - q_start) - duration) < 0.01


def test_quantize_strength_zero_is_noop():
    """Strength 0 should leave notes unchanged."""
    grid = np.array([0.0, 0.5, 1.0])
    q_start, q_end = quantize_note_times(0.12, 0.42, grid, strength=0.0)
    assert q_start == 0.12
    assert q_end == 0.42
