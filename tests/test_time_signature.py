"""Tests for time signature detection."""

import numpy as np
import pytest

from transcription.rhythm import (
    TimeSignature,
    detect_time_signature,
    _score_grouping,
)


class TestTimeSignature:
    def test_str_representation(self):
        ts = TimeSignature(numerator=3, denominator=4, confidence=0.8)
        assert str(ts) == "3/4"

    def test_default_for_short_audio(self):
        """Very short audio with few beats defaults to 4/4."""
        y = np.zeros(22050, dtype=np.float32)
        beat_times = np.array([0.5, 1.0])  # only 2 beats
        ts = detect_time_signature(y, 22050, beat_times)
        assert ts.numerator == 4
        assert ts.denominator == 4
        assert ts.confidence == 0.0

    def test_score_grouping_strong_downbeats(self):
        """When every Nth beat is stronger, grouping of N should score highest."""
        # Simulate 4/4: beat 0 strong, beats 1-3 weak
        strengths = np.tile([1.0, 0.3, 0.5, 0.3], 8)
        score_4 = _score_grouping(strengths, 4)
        score_3 = _score_grouping(strengths, 3)
        assert score_4 > score_3

    def test_score_grouping_waltz_pattern(self):
        """3/4 waltz: beat 0 strong, beats 1-2 weak."""
        strengths = np.tile([1.0, 0.2, 0.2], 10)
        score_3 = _score_grouping(strengths, 3)
        score_4 = _score_grouping(strengths, 4)
        assert score_3 > score_4

    def test_score_grouping_empty(self):
        strengths = np.array([])
        assert _score_grouping(strengths, 4) == 0.0

    def test_detect_returns_valid_structure(self):
        """Basic check that detect_time_signature returns valid output."""
        sr = 22050
        y = np.random.randn(sr * 3).astype(np.float32) * 0.01
        beat_times = np.linspace(0.5, 2.5, 16)  # 16 evenly spaced beats
        ts = detect_time_signature(y, sr, beat_times)
        assert ts.numerator in (2, 3, 4, 6)
        assert ts.denominator in (4, 8)
        assert 0.0 <= ts.confidence <= 1.0
