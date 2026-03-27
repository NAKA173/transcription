"""Tests for per-stem automatic parameter tuning."""

from transcription.transcriber import STEM_PARAMS, TranscribeParams, _get_stem_params


class TestStemParams:
    def test_all_stems_have_configs(self):
        for stem in ("drums", "bass", "vocals", "other"):
            assert stem in STEM_PARAMS

    def test_drum_params_aggressive(self):
        """Drums should use lower thresholds and shorter min note length."""
        drums = STEM_PARAMS["drums"]
        other = STEM_PARAMS["other"]
        assert drums["onset_threshold"] < other["onset_threshold"]
        assert drums["minimum_note_length_ms"] < other["minimum_note_length_ms"]
        assert drums["melodia_trick"] is False

    def test_bass_params_longer_notes(self):
        """Bass notes tend to be longer, so min note length should be moderate."""
        bass = STEM_PARAMS["bass"]
        drums = STEM_PARAMS["drums"]
        assert bass["minimum_note_length_ms"] > drums["minimum_note_length_ms"]

    def test_vocals_keep_pitch_bends(self):
        """Vocal pitch bends are musically meaningful."""
        assert STEM_PARAMS["vocals"]["remove_pitch_bends"] is False

    def test_get_stem_params_returns_transcribe_params(self):
        user = TranscribeParams(quantize_strength=0.5)
        result = _get_stem_params("drums", user)
        assert isinstance(result, TranscribeParams)
        assert result.onset_threshold == STEM_PARAMS["drums"]["onset_threshold"]
        # User's quantize_strength should be passed through
        assert result.quantize_strength == 0.5

    def test_unknown_stem_uses_other_defaults(self):
        user = TranscribeParams()
        result = _get_stem_params("unknown_stem", user)
        assert result.onset_threshold == STEM_PARAMS["other"]["onset_threshold"]

    def test_preprocess_disabled_in_stem_params(self):
        """Per-stem params disable preprocess (already done at stem level)."""
        user = TranscribeParams(preprocess=True)
        result = _get_stem_params("bass", user)
        assert result.preprocess is False
