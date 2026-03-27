from pathlib import Path

from transcription.transcriber import TranscribeParams, TranscribeResult, transcribe_audio


def test_transcribe_wav_produces_midi(sample_wav: Path):
    """A WAV file with a tone should produce a valid MIDI file."""
    result = transcribe_audio(sample_wav)
    assert result.midi_path.exists()
    assert result.midi_path.suffix == ".mid"
    assert result.midi_path.stat().st_size > 0


def test_transcribe_silent_wav(silent_wav: Path):
    """A silent WAV should still produce a MIDI file (possibly with no notes)."""
    result = transcribe_audio(silent_wav)
    assert result.midi_path.exists()
    assert result.midi_path.suffix == ".mid"


def test_high_velocity_filter_removes_notes(sample_wav: Path):
    """A very high min_velocity filter should produce fewer or zero notes."""
    low_filter = transcribe_audio(
        sample_wav, TranscribeParams(min_velocity=1, quantize_enabled=False),
    )
    high_filter = transcribe_audio(
        sample_wav, TranscribeParams(min_velocity=120, quantize_enabled=False),
    )
    assert high_filter.note_count <= low_filter.note_count


def test_quantize_enabled_returns_bpm(sample_wav: Path):
    """With quantization enabled, detected_bpm should be set (possibly 0 for short clips)."""
    result = transcribe_audio(sample_wav, TranscribeParams(quantize_enabled=True))
    assert result.detected_bpm is not None


def test_quantize_disabled_skips_bpm(sample_wav: Path):
    """With quantization disabled, BPM should be None."""
    result = transcribe_audio(sample_wav, TranscribeParams(quantize_enabled=False))
    assert result.detected_bpm is None
