from pathlib import Path

from transcription.transcriber import TranscribeParams, transcribe_audio


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


def test_clean_preset_produces_fewer_notes(sample_wav: Path):
    """Higher thresholds (clean preset) should produce fewer or equal notes."""
    import pretty_midi

    default_path = transcribe_audio(sample_wav, TranscribeParams())
    clean_path = transcribe_audio(
        sample_wav,
        TranscribeParams(onset_threshold=0.75, frame_threshold=0.65, min_velocity=45),
    )

    default_midi = pretty_midi.PrettyMIDI(str(default_path))
    clean_midi = pretty_midi.PrettyMIDI(str(clean_path))

    default_notes = sum(len(i.notes) for i in default_midi.instruments)
    clean_notes = sum(len(i.notes) for i in clean_midi.instruments)

    assert clean_notes <= default_notes
