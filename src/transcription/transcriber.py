from pathlib import Path

from basic_pitch.inference import predict

from transcription.config import OUTPUT_DIR


def transcribe_audio(audio_path: Path) -> Path:
    """Convert an audio file to MIDI using Basic Pitch.

    Args:
        audio_path: Path to the input audio file (MP3 or WAV).

    Returns:
        Path to the generated MIDI file.
    """
    _model_output, midi_data, _note_events = predict(str(audio_path))

    output_path = OUTPUT_DIR / f"{audio_path.stem}.mid"
    midi_data.write(str(output_path))

    return output_path
