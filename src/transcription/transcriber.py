import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import pretty_midi
from basic_pitch.inference import predict

from transcription.config import OUTPUT_DIR


# Per-stem optimized parameters for instrument separation
STEM_PARAMS = {
    "drums": {
        "onset_threshold": 0.45,
        "frame_threshold": 0.35,
        "minimum_note_length_ms": 60.0,
        "min_velocity": 15,
        "melodia_trick": False,
        "remove_pitch_bends": True,
    },
    "bass": {
        "onset_threshold": 0.55,
        "frame_threshold": 0.45,
        "minimum_note_length_ms": 150.0,
        "min_velocity": 25,
        "melodia_trick": True,
        "remove_pitch_bends": True,
    },
    "vocals": {
        "onset_threshold": 0.5,
        "frame_threshold": 0.4,
        "minimum_note_length_ms": 200.0,
        "min_velocity": 20,
        "melodia_trick": True,
        "remove_pitch_bends": False,  # vocals often have meaningful pitch bends
    },
    "other": {
        "onset_threshold": 0.6,
        "frame_threshold": 0.5,
        "minimum_note_length_ms": 180.0,
        "min_velocity": 30,
        "melodia_trick": True,
        "remove_pitch_bends": True,
    },
}


# Progress callback type: (step_name, progress_pct 0-100)
ProgressCallback = Callable[[str, int], None]


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
    preprocess: bool = True  # audio normalization + noise reduction
    drum_detail: bool = False  # split drums into hi-hat/snare/tom/etc.


@dataclass
class TranscribeResult:
    midi_path: Path
    detected_bpm: float | None = None
    note_count: int = 0
    stems_used: list[str] = field(default_factory=list)
    time_signature: str = ""
    drum_parts: list[str] = field(default_factory=list)


def transcribe_audio(
    audio_path: Path,
    params: TranscribeParams | None = None,
    on_progress: ProgressCallback | None = None,
) -> TranscribeResult:
    """Convert an audio file to MIDI using Basic Pitch.

    If separate_instruments is enabled, uses Demucs to separate the audio
    into stems (vocals, drums, bass, other) and transcribes each independently,
    producing a multi-track MIDI file.
    """
    if params is None:
        params = TranscribeParams()
    if on_progress is None:
        on_progress = lambda step, pct: None

    if params.separate_instruments:
        return _transcribe_with_separation(audio_path, params, on_progress)
    else:
        return _transcribe_single(audio_path, params, on_progress)


def _transcribe_single(
    audio_path: Path, params: TranscribeParams, on_progress: ProgressCallback,
) -> TranscribeResult:
    """Single-track transcription (original behavior)."""
    preprocessed = None
    input_path = audio_path

    try:
        # Preprocessing
        if params.preprocess:
            on_progress("前処理中（ノーマライズ・ノイズ除去）", 5)
            from transcription.preprocess import preprocess_audio
            preprocessed = preprocess_audio(audio_path)
            input_path = preprocessed

        on_progress("MIDI変換中", 20)
        _model_output, midi_data, _note_events = predict(
            str(input_path),
            onset_threshold=params.onset_threshold,
            frame_threshold=params.frame_threshold,
            minimum_note_length=params.minimum_note_length_ms,
            melodia_trick=params.melodia_trick,
        )

        on_progress("後処理中", 70)
        midi_data = _postprocess_midi(midi_data, params)

        detected_bpm = None
        time_sig_str = ""
        if params.quantize_enabled and params.quantize_strength > 0:
            on_progress("リズム解析中", 80)
            detected_bpm, time_sig_str = _apply_quantization(
                midi_data, audio_path, params,
            )

        drum_parts = []
        if params.drum_detail:
            from transcription.drums import separate_drum_notes
            new_instruments = []
            for inst in midi_data.instruments:
                if inst.is_drum and inst.notes:
                    sep = separate_drum_notes(inst)
                    for part_name, part_inst in sep.instruments.items():
                        new_instruments.append(part_inst)
                        drum_parts.append(part_name)
                else:
                    new_instruments.append(inst)
            midi_data.instruments = new_instruments

        note_count = sum(len(i.notes) for i in midi_data.instruments)

        output_path = OUTPUT_DIR / f"{audio_path.stem}_{uuid.uuid4().hex[:8]}.mid"

        if time_sig_str:
            _write_time_signature(midi_data, time_sig_str)

        midi_data.write(str(output_path))

        on_progress("完了", 100)
        return TranscribeResult(
            midi_path=output_path,
            detected_bpm=detected_bpm,
            note_count=note_count,
            time_signature=time_sig_str,
            drum_parts=drum_parts,
        )
    finally:
        if preprocessed and preprocessed.exists():
            preprocessed.unlink(missing_ok=True)


def _transcribe_with_separation(
    audio_path: Path, params: TranscribeParams, on_progress: ProgressCallback,
) -> TranscribeResult:
    """Multi-track transcription using Demucs source separation."""
    from transcription.separator import (
        STEM_NAMES, STEM_PROGRAMS, cleanup_separation, separate_audio,
    )

    on_progress("楽器分離中（Demucs）", 5)
    sep_result = separate_audio(audio_path)

    try:
        # Preprocess each stem if enabled
        preprocessed_stems = {}
        if params.preprocess:
            on_progress("ステム前処理中", 20)
            from transcription.preprocess import preprocess_audio
            for stem_name, stem_path in sep_result.stems.items():
                preprocessed_stems[stem_name] = preprocess_audio(stem_path)

        # Detect beats from the original mix (drums help most here)
        on_progress("リズム解析中", 30)
        detected_bpm = None
        beat_grid = None
        time_sig_str = ""
        if params.quantize_enabled and params.quantize_strength > 0:
            from transcription.rhythm import detect_beats
            # Prefer drums stem for beat detection if available
            beat_source = sep_result.stems.get("drums", audio_path)
            beat_info = detect_beats(beat_source)
            detected_bpm = beat_info.tempo
            grid = beat_info.subdivisions
            if len(grid) >= 2:
                beat_grid = grid
            if beat_info.time_signature:
                time_sig_str = str(beat_info.time_signature)

        # Build multi-track MIDI
        combined = pretty_midi.PrettyMIDI(initial_tempo=detected_bpm or 120.0)
        stems_used = []

        stem_list = list(sep_result.stems.items())
        for idx, (stem_name, stem_path) in enumerate(stem_list):
            progress_pct = 40 + int(50 * idx / max(len(stem_list), 1))
            on_progress(f"{STEM_NAMES.get(stem_name, stem_name)}を変換中", progress_pct)

            # Use preprocessed stem if available
            actual_path = preprocessed_stems.get(stem_name, stem_path)

            # Get per-stem optimized parameters
            stem_p = _get_stem_params(stem_name, params)

            _model_output, midi_data, _note_events = predict(
                str(actual_path),
                onset_threshold=stem_p.onset_threshold,
                frame_threshold=stem_p.frame_threshold,
                minimum_note_length=stem_p.minimum_note_length_ms,
                melodia_trick=stem_p.melodia_trick,
            )
            midi_data = _postprocess_midi(midi_data, stem_p)

            if stem_name == "drums":
                # Create drum instrument (channel 10 in GM)
                inst = pretty_midi.Instrument(
                    program=0, is_drum=True, name=STEM_NAMES[stem_name],
                )
            else:
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

        drum_parts = []
        if params.drum_detail:
            from transcription.drums import separate_drum_notes
            new_instruments = []
            for inst in combined.instruments:
                if inst.is_drum and inst.notes:
                    sep = separate_drum_notes(inst)
                    for part_name, part_inst in sep.instruments.items():
                        new_instruments.append(part_inst)
                        drum_parts.append(part_name)
                else:
                    new_instruments.append(inst)
            combined.instruments = new_instruments

        note_count = sum(len(i.notes) for i in combined.instruments)

        output_path = OUTPUT_DIR / f"{audio_path.stem}_{uuid.uuid4().hex[:8]}.mid"

        if time_sig_str:
            _write_time_signature(combined, time_sig_str)

        combined.write(str(output_path))

        on_progress("完了", 100)
        return TranscribeResult(
            midi_path=output_path,
            detected_bpm=detected_bpm,
            note_count=note_count,
            stems_used=stems_used,
            time_signature=time_sig_str,
            drum_parts=drum_parts,
        )
    finally:
        # Clean up preprocessed stems
        for p in preprocessed_stems.values():
            if p.exists():
                p.unlink(missing_ok=True)
        cleanup_separation(sep_result)


def _get_stem_params(stem_name: str, user_params: TranscribeParams) -> TranscribeParams:
    """Build optimized params for a specific stem type.

    Uses per-stem defaults from STEM_PARAMS, but respects user overrides
    for quantize_enabled/quantize_strength (handled at the caller level).
    """
    defaults = STEM_PARAMS.get(stem_name, STEM_PARAMS["other"])
    return TranscribeParams(
        onset_threshold=defaults["onset_threshold"],
        frame_threshold=defaults["frame_threshold"],
        minimum_note_length_ms=defaults["minimum_note_length_ms"],
        min_velocity=defaults["min_velocity"],
        melodia_trick=defaults["melodia_trick"],
        remove_pitch_bends=defaults["remove_pitch_bends"],
        quantize_enabled=user_params.quantize_enabled,
        quantize_strength=user_params.quantize_strength,
        preprocess=False,  # already preprocessed at stem level
    )


def _apply_quantization(
    midi_data: pretty_midi.PrettyMIDI, audio_path: Path, params: TranscribeParams,
) -> tuple[float | None, str]:
    """Apply beat detection and quantization. Returns (BPM, time_signature_str)."""
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

    time_sig_str = ""
    if beat_info.time_signature:
        time_sig_str = str(beat_info.time_signature)

    return detected_bpm, time_sig_str


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


def _write_time_signature(midi_data: pretty_midi.PrettyMIDI, time_sig_str: str) -> None:
    """Write time signature into MIDI file."""
    parts = time_sig_str.split("/")
    if len(parts) != 2:
        return
    try:
        numerator = int(parts[0])
        denominator = int(parts[1])
    except ValueError:
        return

    ts = pretty_midi.TimeSignature(numerator, denominator, 0.0)
    midi_data.time_signature_changes = [ts]


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
