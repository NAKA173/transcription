"""Quality comparison tests using the reference MIDI (before.mid).

These tests verify that the transcription pipeline produces output
that is structurally comparable to a known-good reference MIDI.
"""

from pathlib import Path
from collections import Counter

import pretty_midi
import pytest


@pytest.fixture
def ref(reference_midi: Path) -> pretty_midi.PrettyMIDI:
    return pretty_midi.PrettyMIDI(str(reference_midi))


class TestReferenceAnalysis:
    """Sanity checks on the reference MIDI to document expected properties."""

    def test_reference_has_multiple_instruments(self, ref: pretty_midi.PrettyMIDI):
        assert len(ref.instruments) == 5

    def test_reference_note_count(self, ref: pretty_midi.PrettyMIDI):
        total = sum(len(i.notes) for i in ref.instruments)
        assert 600 <= total <= 650  # ~617 notes

    def test_reference_key_is_c_or_g(self, ref: pretty_midi.PrettyMIDI):
        """The reference piece is in C major / G major."""
        all_pitches = [n.pitch % 12 for i in ref.instruments for n in i.notes]
        counts = Counter(all_pitches)
        top2 = [pc for pc, _ in counts.most_common(2)]
        # G=7, C=0
        assert 7 in top2 and 0 in top2

    def test_reference_has_variable_tempo(self, ref: pretty_midi.PrettyMIDI):
        _, tempos = ref.get_tempo_changes()
        assert len(tempos) > 1

    def test_reference_pitch_range(self, ref: pretty_midi.PrettyMIDI):
        all_pitches = [n.pitch for i in ref.instruments for n in i.notes]
        assert min(all_pitches) >= 40  # ~E2
        assert max(all_pitches) <= 90  # ~F#6


class TestPostProcessQuality:
    """Test that postprocessing steps produce cleaner output."""

    def test_pitch_bends_removed(self):
        """Pitch bends should be stripped from output by default."""
        from transcription.transcriber import TranscribeParams, _postprocess_midi

        midi = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
        inst.pitch_bends.append(pretty_midi.PitchBend(pitch=1000, time=0.1))
        midi.instruments.append(inst)

        params = TranscribeParams(remove_pitch_bends=True)
        result = _postprocess_midi(midi, params)

        assert len(result.instruments[0].pitch_bends) == 0
        assert len(result.instruments[0].notes) == 1

    def test_overlapping_notes_merged(self):
        """Overlapping notes on same pitch should be merged."""
        from transcription.transcriber import TranscribeParams, _postprocess_midi

        midi = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
        inst.notes.append(pretty_midi.Note(velocity=70, pitch=60, start=0.3, end=0.8))
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.6, end=1.0))
        midi.instruments.append(inst)

        params = TranscribeParams()
        result = _postprocess_midi(midi, params)

        notes = result.instruments[0].notes
        assert len(notes) == 1
        assert notes[0].start == 0.0
        assert notes[0].end == 1.0
        assert notes[0].velocity == 90  # max of all merged

    def test_quiet_notes_removed(self):
        """Notes below min_velocity threshold should be removed."""
        from transcription.transcriber import TranscribeParams, _postprocess_midi

        midi = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=10, pitch=60, start=0.0, end=0.5))
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0.0, end=0.5))
        midi.instruments.append(inst)

        params = TranscribeParams(min_velocity=30)
        result = _postprocess_midi(midi, params)

        assert len(result.instruments[0].notes) == 1
        assert result.instruments[0].notes[0].pitch == 64
