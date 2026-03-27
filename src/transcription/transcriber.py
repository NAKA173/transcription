from pathlib import Path
from dataclasses import dataclass

import pretty_midi
from basic_pitch.inference import predict

from transcription.config import OUTPUT_DIR


@dataclass
class TranscribeParams:
    onset_threshold: float = 0.6
    frame_threshold: float = 0.5
    minimum_note_length_ms: float = 200.0
    min_velocity: int = 30
    quantize_to_bpm: float | None = None


def transcribe_audio(audio_path: Path, params: TranscribeParams | None = None) -> Path:
    """Convert an audio file to MIDI using Basic Pitch.

    Args:
        audio_path: Path to the input audio file (MP3 or WAV).
        params: Transcription parameters. Uses improved defaults if None.

    Returns:
        Path to the generated MIDI file.
    """
    if params is None:
        params = TranscribeParams()

    _model_output, midi_data, _note_events = predict(
        str(audio_path),
        onset_threshold=params.onset_threshold,
        frame_threshold=params.frame_threshold,
        minimum_note_length=params.minimum_note_length_ms,
    )

    midi_data = _postprocess_midi(midi_data, params)

    output_path = OUTPUT_DIR / f"{audio_path.stem}.mid"
    midi_data.write(str(output_path))

    return output_path


def _postprocess_midi(midi_data: pretty_midi.PrettyMIDI, params: TranscribeParams) -> pretty_midi.PrettyMIDI:
    """Clean up MIDI output to reduce noise and improve musicality."""
    for instrument in midi_data.instruments:
        # Remove very quiet notes (likely false detections)
        instrument.notes = [n for n in instrument.notes if n.velocity >= params.min_velocity]

        # Remove overlapping duplicate notes on the same pitch
        instrument.notes.sort(key=lambda n: (n.pitch, n.start))
        cleaned = []
        for note in instrument.notes:
            if cleaned and cleaned[-1].pitch == note.pitch:
                prev = cleaned[-1]
                # If this note starts before the previous ends, merge or skip
                if note.start < prev.end:
                    prev.end = max(prev.end, note.end)
                    prev.velocity = max(prev.velocity, note.velocity)
                    continue
            cleaned.append(note)
        instrument.notes = cleaned

        # Sort by start time for clean output
        instrument.notes.sort(key=lambda n: n.start)

    return midi_data
