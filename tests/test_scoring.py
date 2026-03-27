"""Tests for MIDI accuracy scoring."""

from pathlib import Path

import pretty_midi
import pytest

from transcription.scoring import score_midi, ScoreResult


def _make_midi(notes: list[tuple[int, float, float, int]], path: Path) -> Path:
    """Create a simple MIDI file with given notes: (pitch, start, end, velocity)."""
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end, vel in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=vel, pitch=pitch, start=start, end=end,
        ))
    midi.instruments.append(inst)
    midi.write(str(path))
    return path


class TestScoring:
    def test_identical_midi_perfect_score(self, tmp_path: Path):
        notes = [(60, 0.0, 0.5, 100), (64, 0.5, 1.0, 90), (67, 1.0, 1.5, 80)]
        ref = _make_midi(notes, tmp_path / "ref.mid")
        pred = _make_midi(notes, tmp_path / "pred.mid")

        result = score_midi(ref, pred)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.matched_notes == 3

    def test_no_overlap_zero_score(self, tmp_path: Path):
        ref_notes = [(60, 0.0, 0.5, 100)]
        pred_notes = [(80, 5.0, 5.5, 100)]  # completely different
        ref = _make_midi(ref_notes, tmp_path / "ref.mid")
        pred = _make_midi(pred_notes, tmp_path / "pred.mid")

        result = score_midi(ref, pred)
        assert result.f1 == 0.0
        assert result.matched_notes == 0

    def test_partial_match(self, tmp_path: Path):
        ref_notes = [(60, 0.0, 0.5, 100), (64, 1.0, 1.5, 100)]
        pred_notes = [(60, 0.02, 0.52, 90)]  # matches first, misses second
        ref = _make_midi(ref_notes, tmp_path / "ref.mid")
        pred = _make_midi(pred_notes, tmp_path / "pred.mid")

        result = score_midi(ref, pred, onset_tolerance=0.1, offset_tolerance=0.1)
        assert result.matched_notes == 1
        assert result.precision == 1.0  # 1/1 predicted matched
        assert result.recall == 0.5     # 1/2 reference matched

    def test_onset_tolerance(self, tmp_path: Path):
        ref_notes = [(60, 1.0, 1.5, 100)]
        pred_notes = [(60, 1.15, 1.65, 100)]  # 0.15s off
        ref = _make_midi(ref_notes, tmp_path / "ref.mid")
        pred = _make_midi(pred_notes, tmp_path / "pred.mid")

        # With tight tolerance, no match
        tight = score_midi(ref, pred, onset_tolerance=0.05, offset_tolerance=0.3)
        assert tight.matched_notes == 0

        # With loose tolerance, match
        loose = score_midi(ref, pred, onset_tolerance=0.2, offset_tolerance=0.3)
        assert loose.matched_notes == 1

    def test_empty_reference(self, tmp_path: Path):
        ref = _make_midi([], tmp_path / "ref.mid")
        pred = _make_midi([(60, 0.0, 0.5, 100)], tmp_path / "pred.mid")
        result = score_midi(ref, pred)
        assert result.f1 == 0.0

    def test_empty_prediction(self, tmp_path: Path):
        ref = _make_midi([(60, 0.0, 0.5, 100)], tmp_path / "ref.mid")
        pred = _make_midi([], tmp_path / "pred.mid")
        result = score_midi(ref, pred)
        assert result.f1 == 0.0

    def test_onset_mae(self, tmp_path: Path):
        ref_notes = [(60, 1.0, 1.5, 100), (64, 2.0, 2.5, 100)]
        pred_notes = [(60, 1.05, 1.55, 100), (64, 1.97, 2.47, 100)]
        ref = _make_midi(ref_notes, tmp_path / "ref.mid")
        pred = _make_midi(pred_notes, tmp_path / "pred.mid")

        result = score_midi(ref, pred, onset_tolerance=0.1, offset_tolerance=0.2)
        assert result.matched_notes == 2
        assert 0.03 < result.onset_mae < 0.06  # average of 0.05 and 0.03


class TestScoringWithFixtures:
    """Score test.mid against itself (sanity check)."""

    def test_self_comparison_perfect(self, test_midi: Path):
        result = score_midi(test_midi, test_midi)
        assert result.f1 == 1.0
        assert result.matched_notes == result.total_ref_notes
