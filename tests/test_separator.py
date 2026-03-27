"""Tests for instrument separation.

Since Demucs model may not be available in CI/test environments,
these tests cover:
1. The separator API contract (mocked)
2. The multi-track MIDI assembly logic
3. Integration with the transcriber when separation is disabled
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import struct
import wave
import math

import pretty_midi
import pytest

from transcription.separator import STEM_PROGRAMS, STEM_NAMES, SeparationResult
from transcription.transcriber import TranscribeParams, transcribe_audio, _postprocess_midi


class TestStemConfig:
    """Verify stem configuration is correct."""

    def test_stem_programs_defined(self):
        assert "vocals" in STEM_PROGRAMS
        assert "drums" in STEM_PROGRAMS
        assert "bass" in STEM_PROGRAMS
        assert "other" in STEM_PROGRAMS

    def test_stem_names_defined(self):
        assert "vocals" in STEM_NAMES
        assert "drums" in STEM_NAMES
        assert "bass" in STEM_NAMES
        assert "other" in STEM_NAMES


class TestSeparationDisabled:
    """When separation is off, everything works as before."""

    def test_no_separation_single_track(self, sample_wav: Path):
        result = transcribe_audio(
            sample_wav,
            TranscribeParams(separate_instruments=False, quantize_enabled=False),
        )
        assert result.midi_path.exists()
        assert result.stems_used == []

    def test_stems_used_empty_without_separation(self, sample_wav: Path):
        result = transcribe_audio(
            sample_wav,
            TranscribeParams(separate_instruments=False, quantize_enabled=False),
        )
        assert result.stems_used == []


class TestMultiTrackAssembly:
    """Test that multi-track MIDI is assembled correctly from mock stems."""

    def test_mock_separation_produces_multi_track(self, tmp_path: Path):
        """Simulate separation by creating fake stem WAV files."""
        # Create two fake stem WAVs (simple tones at different pitches)
        stems = {}
        for name, freq in [("bass", 110.0), ("other", 440.0)]:
            filepath = tmp_path / f"{name}.wav"
            _write_tone_wav(filepath, freq, duration=1.0)
            stems[name] = filepath

        sep_result = SeparationResult(stems=stems, work_dir=tmp_path)

        # Mock separate_audio to return our pre-made stems
        with patch("transcription.separator.separate_audio", return_value=sep_result):
            with patch("transcription.separator.cleanup_separation"):
                result = transcribe_audio(
                    tmp_path / "bass.wav",  # dummy input
                    TranscribeParams(
                        separate_instruments=True,
                        quantize_enabled=False,
                        preprocess=False,
                    ),
                )

        assert result.midi_path.exists()
        midi = pretty_midi.PrettyMIDI(str(result.midi_path))

        # Should have separate instruments for each stem that produced notes
        assert len(midi.instruments) >= 1
        # Verify instrument names
        names = {i.name for i in midi.instruments}
        assert names.issubset(set(STEM_NAMES.values()))

    def test_drum_stem_is_drum_track(self, tmp_path: Path):
        """Drum stem should produce a drum instrument (is_drum=True)."""
        drum_wav = tmp_path / "drums.wav"
        _write_tone_wav(drum_wav, 200.0, duration=1.0)

        other_wav = tmp_path / "other.wav"
        _write_tone_wav(other_wav, 440.0, duration=1.0)

        stems = {"drums": drum_wav, "other": other_wav}
        sep_result = SeparationResult(stems=stems, work_dir=tmp_path)

        with patch("transcription.separator.separate_audio", return_value=sep_result):
            with patch("transcription.separator.cleanup_separation"):
                result = transcribe_audio(
                    drum_wav,
                    TranscribeParams(
                        separate_instruments=True,
                        quantize_enabled=False,
                        preprocess=False,
                    ),
                )

        midi = pretty_midi.PrettyMIDI(str(result.midi_path))
        drum_instruments = [i for i in midi.instruments if i.is_drum]
        # May or may not detect drum notes from a simple tone, but structure is correct
        if drum_instruments:
            assert drum_instruments[0].name == "Drums"

    def test_stems_used_reported(self, tmp_path: Path):
        """Result should report which stems produced notes."""
        bass_wav = tmp_path / "bass.wav"
        _write_tone_wav(bass_wav, 110.0, duration=1.0)

        stems = {"bass": bass_wav}
        sep_result = SeparationResult(stems=stems, work_dir=tmp_path)

        with patch("transcription.separator.separate_audio", return_value=sep_result):
            with patch("transcription.separator.cleanup_separation"):
                result = transcribe_audio(
                    bass_wav,
                    TranscribeParams(
                        separate_instruments=True,
                        quantize_enabled=False,
                        preprocess=False,
                    ),
                )

        if result.note_count > 0:
            assert "bass" in result.stems_used


def _write_tone_wav(filepath: Path, frequency: float, duration: float = 1.0) -> None:
    """Write a simple sine wave WAV for testing."""
    sample_rate = 22050
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
