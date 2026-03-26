from pathlib import Path

from transcription.transcriber import transcribe_audio


def test_transcribe_wav_produces_midi(sample_wav: Path):
    """A WAV file with a tone should produce a valid MIDI file."""
    midi_path = transcribe_audio(sample_wav)
    assert midi_path.exists()
    assert midi_path.suffix == ".mid"
    assert midi_path.stat().st_size > 0


def test_transcribe_silent_wav(silent_wav: Path):
    """A silent WAV should still produce a MIDI file (possibly with no notes)."""
    midi_path = transcribe_audio(silent_wav)
    assert midi_path.exists()
    assert midi_path.suffix == ".mid"
