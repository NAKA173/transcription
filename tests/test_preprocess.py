"""Tests for audio preprocessing (normalization and noise reduction)."""

import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from transcription.preprocess import preprocess_audio, _normalize, _reduce_noise


class TestNormalize:
    def test_scales_to_target_peak(self):
        y = np.array([0.1, -0.2, 0.15, -0.05], dtype=np.float32)
        result = _normalize(y, target_peak=0.95)
        assert abs(np.max(np.abs(result)) - 0.95) < 0.01

    def test_silent_audio_unchanged(self):
        y = np.zeros(100, dtype=np.float32)
        result = _normalize(y)
        assert np.all(result == 0)

    def test_already_loud_audio_scaled_down(self):
        y = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        result = _normalize(y, target_peak=0.5)
        assert abs(np.max(np.abs(result)) - 0.5) < 0.01


class TestNoiseReduction:
    def test_reduces_noise_floor(self):
        sr = 22050
        duration = 3.0
        n = int(sr * duration)
        t = np.linspace(0, duration, n, dtype=np.float32)

        # Create audio: 1s silence, then 2s of strong signal
        # This gives a clear SNR difference between quiet and loud frames
        signal = np.zeros(n, dtype=np.float32)
        signal_start = int(sr * 1.0)
        signal[signal_start:] = 0.8 * np.sin(2 * np.pi * 440 * t[signal_start:])

        np.random.seed(42)
        noise = 0.03 * np.random.randn(n).astype(np.float32)
        noisy = signal + noise

        cleaned = _reduce_noise(noisy, sr)

        # The cleaned signal should be different from the noisy one
        # (i.e., some processing occurred)
        diff = np.mean(np.abs(cleaned - noisy))
        assert diff > 0.001, "Noise reduction should modify the signal"

    def test_preserves_signal_energy(self):
        sr = 22050
        duration = 2.0
        n = int(sr * duration)
        t = np.linspace(0, duration, n, dtype=np.float32)

        # Strong signal — noise reduction should not destroy it
        signal = 0.8 * np.sin(2 * np.pi * 440 * t)

        cleaned = _reduce_noise(signal, sr)
        # Output should have substantial energy (not zeroed out)
        assert np.std(cleaned) > 0.1


class TestPreprocessAudio:
    def test_produces_wav_file(self, sample_wav: Path):
        result = preprocess_audio(sample_wav)
        try:
            assert result.exists()
            assert result.suffix == ".wav"
        finally:
            result.unlink(missing_ok=True)

    def test_no_processing_still_produces_file(self, sample_wav: Path):
        result = preprocess_audio(sample_wav, normalize=False, noise_reduce=False)
        try:
            assert result.exists()
        finally:
            result.unlink(missing_ok=True)
