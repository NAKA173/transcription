import pytest
import pretty_midi

from transcription.drums import (
    GM_DRUM_MAP,
    DRUM_PART_ORDER,
    classify_drum_note,
    separate_drum_notes,
)


def test_classify_known_gm_notes():
    assert classify_drum_note(36) == "kick"
    assert classify_drum_note(38) == "snare"
    assert classify_drum_note(42) == "hihat"
    assert classify_drum_note(45) == "tom"
    assert classify_drum_note(49) == "crash"
    assert classify_drum_note(51) == "ride"


def test_classify_fallback_heuristic():
    result = classify_drum_note(30)
    assert result == "kick"
    result = classify_drum_note(60)
    assert result == "crash"


def test_separate_drum_notes_basic():
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=36, start=0.0, end=0.1))
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=38, start=0.5, end=0.6))
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=42, start=1.0, end=1.1))
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=45, start=1.5, end=1.6))
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=49, start=2.0, end=2.1))
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=51, start=2.5, end=2.6))

    result = separate_drum_notes(inst)
    assert "kick" in result.instruments
    assert "snare" in result.instruments
    assert "hihat" in result.instruments
    assert "tom" in result.instruments
    assert "crash" in result.instruments
    assert "ride" in result.instruments
    assert result.note_counts["kick"] == 1
    assert result.note_counts["snare"] == 1


def test_separate_empty_drum():
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    result = separate_drum_notes(inst)
    assert len(result.instruments) == 0


def test_separate_preserves_all_notes():
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for pitch in [36, 36, 38, 42, 42, 42, 45, 49, 51]:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=0.1))

    result = separate_drum_notes(inst)
    total = sum(result.note_counts.values())
    assert total == 9


def test_gm_drum_map_covers_common_notes():
    common = [35, 36, 38, 40, 42, 44, 46, 41, 43, 45, 47, 48, 49, 51, 52]
    for note in common:
        assert note in GM_DRUM_MAP


def test_drum_part_order():
    assert "kick" in DRUM_PART_ORDER
    assert "snare" in DRUM_PART_ORDER
    assert "hihat" in DRUM_PART_ORDER
    assert "ride" in DRUM_PART_ORDER
