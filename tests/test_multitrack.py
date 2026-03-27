"""Tests using test.mid — a multi-section, multi-instrument reference file.

Sections (approximate):
  0-35s:  Grand Piano + Electric Piano (piano + chords)
  35-65s: Grand Piano + Electric Piano + Drums (piano + chords + drums)
  65-80s: Electric Piano only (chords only)
  80-110s: Grand Piano only (piano only)
"""

from pathlib import Path

import pretty_midi
import pytest


class TestMultiTrackStructure:
    """Verify test.mid structure matches expected layout."""

    def test_has_three_instruments(self, test_midi: Path):
        midi = pretty_midi.PrettyMIDI(str(test_midi))
        assert len(midi.instruments) == 3

    def test_instrument_types(self, test_midi: Path):
        midi = pretty_midi.PrettyMIDI(str(test_midi))
        names = {i.name for i in midi.instruments}
        assert "Grand Piano" in names
        assert "Electric Piano" in names

        drum_instruments = [i for i in midi.instruments if i.is_drum]
        assert len(drum_instruments) == 1
        assert drum_instruments[0].name == "2023 Drum Kit"

    def test_tempo_is_110(self, test_midi: Path):
        midi = pretty_midi.PrettyMIDI(str(test_midi))
        tempos = midi.get_tempo_changes()
        assert len(tempos[1]) >= 1
        assert abs(tempos[1][0] - 110.0) < 1.0

    def test_note_counts(self, test_midi: Path):
        midi = pretty_midi.PrettyMIDI(str(test_midi))
        by_name = {i.name: len(i.notes) for i in midi.instruments}
        assert by_name["Grand Piano"] == 261
        assert by_name["Electric Piano"] == 142
        # Drum kit
        drum = [i for i in midi.instruments if i.is_drum][0]
        assert len(drum.notes) == 32

    def test_total_duration(self, test_midi: Path):
        midi = pretty_midi.PrettyMIDI(str(test_midi))
        assert 100.0 < midi.get_end_time() < 115.0


class TestSectionDetection:
    """Verify the four sections are distinguishable by instrument activity."""

    @staticmethod
    def _active_instruments(midi, start, end):
        """Return set of instrument names with notes in [start, end)."""
        active = set()
        for inst in midi.instruments:
            for note in inst.notes:
                if note.start < end and note.end > start:
                    active.add(inst.name if not inst.is_drum else "drums")
                    break
        return active

    def test_section_piano_and_chords(self, test_midi: Path):
        """0-30s: Piano + Electric Piano, no drums."""
        midi = pretty_midi.PrettyMIDI(str(test_midi))
        active = self._active_instruments(midi, 5.0, 30.0)
        assert "Grand Piano" in active
        assert "Electric Piano" in active
        assert "drums" not in active

    def test_section_with_drums(self, test_midi: Path):
        """40-60s: All three instruments active."""
        midi = pretty_midi.PrettyMIDI(str(test_midi))
        active = self._active_instruments(midi, 40.0, 60.0)
        assert "Grand Piano" in active
        assert "Electric Piano" in active
        assert "drums" in active

    def test_section_chords_only(self, test_midi: Path):
        """70-78s: Only Electric Piano."""
        midi = pretty_midi.PrettyMIDI(str(test_midi))
        active = self._active_instruments(midi, 70.0, 78.0)
        assert "Electric Piano" in active
        assert "Grand Piano" not in active
        assert "drums" not in active

    def test_section_piano_only(self, test_midi: Path):
        """90-108s: Only Grand Piano."""
        midi = pretty_midi.PrettyMIDI(str(test_midi))
        active = self._active_instruments(midi, 90.0, 108.0)
        assert "Grand Piano" in active
        assert "Electric Piano" not in active
        assert "drums" not in active
