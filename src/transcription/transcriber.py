import uuid
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
    quantize_enabled: bool = True
    quantize_strength: float = 0.8  # 0.0 = off, 1.0 = full snap
    remove_pitch_bends: bool = True
    melodia_trick: bool = True


@dataclass
class TranscribeResult:
    midi_path: Path
    detected_bpm: float | None = None
    note_count: int = 0


def transcribe_audio(audio_path: Path, params: TranscribeParams | None = None) -> TranscribeResult:
    """Convert an audio file to MIDI using Basic Pitch.

    Args:
        audio_path: Path to the input audio file (MP3 or WAV).
        params: Transcription parameters. Uses improved defaults if None.

    Returns:
        TranscribeResult with MIDI path and detected info.
    """
    if params is None:
        params = TranscribeParams()

    _model_output, midi_data, _note_events = predict(
        str(audio_path),
        onset_threshold=params.onset_threshold,
        frame_threshold=params.frame_threshold,
        minimum_note_length=params.minimum_note_length_ms,
        melodia_trick=params.melodia_trick,
    )

    midi_data = _postprocess_midi(midi_data, params)

    # Beat detection and quantization
    detected_bpm = None
    if params.quantize_enabled and params.quantize_strength > 0:
        from transcription.rhythm import detect_beats, quantize_note_times

        beat_info = detect_beats(audio_path)
        detected_bpm = beat_info.tempo

        grid = beat_info.subdivisions  # 16th note grid
        if len(grid) >= 2:
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    note.start, note.end = quantize_note_times(
                        note.start, note.end, grid, params.quantize_strength,
                    )

            # Apply detected tempo changes to MIDI
            _apply_tempo_map(midi_data, beat_info)

    note_count = sum(len(i.notes) for i in midi_data.instruments)

    output_path = OUTPUT_DIR / f"{audio_path.stem}_{uuid.uuid4().hex[:8]}.mid"
    midi_data.write(str(output_path))

    return TranscribeResult(
        midi_path=output_path,
        detected_bpm=detected_bpm,
        note_count=note_count,
    )


def _apply_tempo_map(midi_data: pretty_midi.PrettyMIDI, beat_info) -> None:
    """Write detected tempo changes into MIDI tempo map.

    Instead of a flat 120 BPM, embed the actual detected tempo so
    DAWs display correct bar/beat positions.
    """
    from transcription.rhythm import BeatInfo

    changes = beat_info.tempo_changes
    if not changes:
        return

    # Clear existing tempo changes and rebuild
    # PrettyMIDI stores tempo as a list of (time, tempo) internally
    # We rebuild the initial tempo change list
    midi_data._tick_scales = []
    midi_data._tick_scales.append(
        (0, 60.0 / (changes[0][1] if changes[0][1] > 0 else 120.0) / midi_data.resolution)
    )

    for time, bpm in changes[1:]:
        if bpm > 0:
            tick = midi_data.time_to_tick(time)
            midi_data._tick_scales.append(
                (tick, 60.0 / bpm / midi_data.resolution)
            )


def _postprocess_midi(midi_data: pretty_midi.PrettyMIDI, params: TranscribeParams) -> pretty_midi.PrettyMIDI:
    """Clean up MIDI output to reduce noise and improve musicality."""
    for instrument in midi_data.instruments:
        # Remove pitch bends (Basic Pitch adds these but they cause artifacts)
        if params.remove_pitch_bends:
            instrument.pitch_bends = []

        # Remove very quiet notes (likely false detections)
        instrument.notes = [n for n in instrument.notes if n.velocity >= params.min_velocity]

        # Remove overlapping duplicate notes on the same pitch
        instrument.notes.sort(key=lambda n: (n.pitch, n.start))
        cleaned = []
        for note in instrument.notes:
            if cleaned and cleaned[-1].pitch == note.pitch:
                prev = cleaned[-1]
                if note.start < prev.end:
                    prev.end = max(prev.end, note.end)
                    prev.velocity = max(prev.velocity, note.velocity)
                    continue
            cleaned.append(note)
        instrument.notes = cleaned

        # Sort by start time for clean output
        instrument.notes.sort(key=lambda n: n.start)

    return midi_data
