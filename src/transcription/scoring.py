"""MIDI accuracy scoring — compare transcription output against a reference.

Computes precision, recall, and F1 for note detection using
tolerances for pitch (exact), onset time, and offset time.
"""

from dataclasses import dataclass
from pathlib import Path

import pretty_midi
import numpy as np


@dataclass
class ScoreResult:
    precision: float      # correctly detected / total detected
    recall: float         # correctly detected / total in reference
    f1: float             # harmonic mean of precision and recall
    pitch_accuracy: float  # % of matched notes with correct pitch
    onset_mae: float      # mean absolute onset error (seconds) for matched notes
    total_ref_notes: int
    total_pred_notes: int
    matched_notes: int


def score_midi(
    reference_path: Path | str,
    predicted_path: Path | str,
    onset_tolerance: float = 0.1,   # seconds
    offset_tolerance: float = 0.2,  # seconds
    pitch_tolerance: int = 0,       # semitones (0 = exact match)
) -> ScoreResult:
    """Compare predicted MIDI against a reference and compute accuracy metrics.

    A predicted note is considered a "match" if:
    - pitch is within pitch_tolerance semitones of a reference note
    - onset is within onset_tolerance seconds
    - offset is within offset_tolerance seconds

    Each reference note can be matched at most once (greedy matching
    by onset proximity).
    """
    ref_midi = pretty_midi.PrettyMIDI(str(reference_path))
    pred_midi = pretty_midi.PrettyMIDI(str(predicted_path))

    ref_notes = _extract_all_notes(ref_midi)
    pred_notes = _extract_all_notes(pred_midi)

    if not ref_notes or not pred_notes:
        return ScoreResult(
            precision=0.0, recall=0.0, f1=0.0,
            pitch_accuracy=0.0, onset_mae=0.0,
            total_ref_notes=len(ref_notes),
            total_pred_notes=len(pred_notes),
            matched_notes=0,
        )

    # Greedy matching: for each predicted note, find best unmatched reference
    ref_matched = set()
    matches = []  # list of (pred_note, ref_note) pairs

    # Sort predicted notes by onset for deterministic matching
    pred_notes.sort(key=lambda n: (n.start, n.pitch))

    for pred in pred_notes:
        best_ref_idx = None
        best_onset_diff = float("inf")

        for j, ref in enumerate(ref_notes):
            if j in ref_matched:
                continue
            if abs(pred.pitch - ref.pitch) > pitch_tolerance:
                continue
            onset_diff = abs(pred.start - ref.start)
            if onset_diff > onset_tolerance:
                continue
            offset_diff = abs(pred.end - ref.end)
            if offset_diff > offset_tolerance:
                continue
            if onset_diff < best_onset_diff:
                best_onset_diff = onset_diff
                best_ref_idx = j

        if best_ref_idx is not None:
            ref_matched.add(best_ref_idx)
            matches.append((pred, ref_notes[best_ref_idx]))

    matched = len(matches)
    precision = matched / len(pred_notes) if pred_notes else 0.0
    recall = matched / len(ref_notes) if ref_notes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Pitch accuracy among matched notes
    pitch_correct = sum(1 for p, r in matches if p.pitch == r.pitch)
    pitch_accuracy = pitch_correct / matched if matched > 0 else 0.0

    # Mean onset error
    onset_errors = [abs(p.start - r.start) for p, r in matches]
    onset_mae = float(np.mean(onset_errors)) if onset_errors else 0.0

    return ScoreResult(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        pitch_accuracy=round(pitch_accuracy, 4),
        onset_mae=round(onset_mae, 4),
        total_ref_notes=len(ref_notes),
        total_pred_notes=len(pred_notes),
        matched_notes=matched,
    )


def _extract_all_notes(midi: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    """Extract all notes from all instruments, flattened into one list."""
    notes = []
    for inst in midi.instruments:
        notes.extend(inst.notes)
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes
