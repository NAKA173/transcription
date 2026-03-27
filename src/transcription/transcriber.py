import uuid
from pathlib import Path
from dataclasses import dataclass, field

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
    separate_instruments: bool = False  # Demucs source separation


@dataclass
class TranscribeResult:
    midi_path: Path
    detected_bpm: float | None = None
    note_count: int = 0
    stems_used: list[str] = field(default_factory=list)


def transcribe_audio(audio_path: Path, params: TranscribeParams | None = None) -> TranscribeResult:
    """Convert an audio file to MIDI using Basic Pitch.

    If separate_instruments is enabled, uses Demucs to separate the audio
    into stems (vocals, drums, bass, other) and transcribes each independently,
    producing a multi-track MIDI file.
    """
    if params is None:
        params = TranscribeParams()

    if params.separate_instruments:
        return _transcribe_with_separation(audio_path, params)
    else:
        return _transcribe_single(audio_path, params)


def _transcribe_single(audio_path: Path, params: TranscribeParams) -> TranscribeResult:
    """Single-track transcription (original behavior)."""
    _model_output, midi_data, _note_events = predict(
        str(audio_path),
        onset_threshold=params.onset_threshold,
        frame_threshold=params.frame_threshold,
        minimum_note_length=params.minimum_note_length_ms,
        melodia_trick=params.melodia_trick,
    )

    midi_data = _postprocess_midi(midi_data, params)

    detected_bpm = None
    if params.quantize_enabled and params.quantize_strength > 0:
        detected_bpm = _apply_quantization(midi_data, audio_path, params)

    note_count = sum(len(i.notes) for i in midi_data.instruments)

    output_path = OUTPUT_DIR / f"{audio_path.stem}_{uuid.uuid4().hex[:8]}.mid"
    midi_data.write(str(output_path))

    return TranscribeResult(
        midi_path=output_path,
        detected_bpm=detected_bpm,
        note_count=note_count,
    )


def _transcribe_with_separation(audio_path: Path, params: TranscribeParams) -> TranscribeResult:
    """Multi-track transcription using Demucs source separation."""
    from transcription.separator import (
        STEM_NAMES, STEM_PROGRAMS, cleanup_separation, separate_audio,
    )

    sep_result = separate_audio(audio_path)

    try:
        # Detect beats from the original mix (drums help most here)
        detected_bpm = None
        beat_grid = None
        if params.quantize_enabled and params.quantize_strength > 0:
            from transcription.rhythm import detect_beats
            # Prefer drums stem for beat detection if available
            beat_source = sep_result.stems.get("drums", audio_path)
            beat_info = detect_beats(beat_source)
            detected_bpm = beat_info.tempo
            grid = beat_info.subdivisions
            if len(grid) >= 2:
                beat_grid = grid

        # Build multi-track MIDI
        combined = pretty_midi.PrettyMIDI(initial_tempo=detected_bpm or 120.0)
        stems_used = []

        for stem_name, stem_path in sep_result.stems.items():
            if stem_name == "drums":
                # Drums: transcribe with more aggressive settings
                drum_params = TranscribeParams(
                    onset_threshold=0.5,
                    frame_threshold=0.4,
                    minimum_note_length_ms=80.0,
                    min_velocity=20,
                    remove_pitch_bends=True,
                    melodia_trick=False,
                )
                _model_output, midi_data, _note_events = predict(
                    str(stem_path),
                    onset_threshold=drum_params.onset_threshold,
                    frame_threshold=drum_params.frame_threshold,
                    minimum_note_length=drum_params.minimum_note_length_ms,
                    melodia_trick=drum_params.melodia_trick,
                )
                midi_data = _postprocess_midi(midi_data, drum_params)

                # Create drum instrument (channel 10 in GM)
                drum_inst = pretty_midi.Instrument(
                    program=0, is_drum=True, name=STEM_NAMES[stem_name],
                )
                for inst in midi_data.instruments:
                    for note in inst.notes:
                        if beat_grid is not None:
                            from transcription.rhythm import quantize_note_times
                            note.start, note.end = quantize_note_times(
                                note.start, note.end, beat_grid, params.quantize_strength,
                            )
                        drum_inst.notes.append(note)
                if drum_inst.notes:
                    combined.instruments.append(drum_inst)
                    stems_used.append(stem_name)

            else:
                # Melodic stems: use normal transcription
                _model_output, midi_data, _note_events = predict(
                    str(stem_path),
                    onset_threshold=params.onset_threshold,
                    frame_threshold=params.frame_threshold,
                    minimum_note_length=params.minimum_note_length_ms,
                    melodia_trick=params.melodia_trick,
                )
                midi_data = _postprocess_midi(midi_data, params)

                program = STEM_PROGRAMS.get(stem_name, 0)
                inst = pretty_midi.Instrument(
                    program=program, name=STEM_NAMES.get(stem_name, stem_name),
                )
                for orig_inst in midi_data.instruments:
                    for note in orig_inst.notes:
                        if beat_grid is not None:
                            from transcription.rhythm import quantize_note_times
                            note.start, note.end = quantize_note_times(
                                note.start, note.end, beat_grid, params.quantize_strength,
                            )
                        inst.notes.append(note)

                inst.notes.sort(key=lambda n: n.start)
                if inst.notes:
                    combined.instruments.append(inst)
                    stems_used.append(stem_name)

        note_count = sum(len(i.notes) for i in combined.instruments)

        output_path = OUTPUT_DIR / f"{audio_path.stem}_{uuid.uuid4().hex[:8]}.mid"
        combined.write(str(output_path))

        return TranscribeResult(
            midi_path=output_path,
            detected_bpm=detected_bpm,
            note_count=note_count,
            stems_used=stems_used,
        )
    finally:
        cleanup_separation(sep_result)


def _apply_quantization(
    midi_data: pretty_midi.PrettyMIDI, audio_path: Path, params: TranscribeParams,
) -> float | None:
    """Apply beat detection and quantization. Returns detected BPM."""
    from transcription.rhythm import detect_beats, quantize_note_times

    beat_info = detect_beats(audio_path)
    detected_bpm = beat_info.tempo

    grid = beat_info.subdivisions
    if len(grid) >= 2:
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                note.start, note.end = quantize_note_times(
                    note.start, note.end, grid, params.quantize_strength,
                )
        _apply_tempo_map(midi_data, beat_info)

    return detected_bpm


def _apply_tempo_map(midi_data: pretty_midi.PrettyMIDI, beat_info) -> None:
    """Write detected tempo changes into MIDI tempo map."""
    changes = beat_info.tempo_changes
    if not changes:
        return

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


def _postprocess_midi(
    midi_data: pretty_midi.PrettyMIDI, params: TranscribeParams,
) -> pretty_midi.PrettyMIDI:
    """Clean up MIDI output to reduce noise and improve musicality."""
    for instrument in midi_data.instruments:
        if params.remove_pitch_bends:
            instrument.pitch_bends = []

        instrument.notes = [n for n in instrument.notes if n.velocity >= params.min_velocity]

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
        instrument.notes.sort(key=lambda n: n.start)

    return midi_data
