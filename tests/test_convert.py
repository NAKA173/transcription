import pytest
from pathlib import Path

from transcription.convert import convert_to_wav, NEEDS_CONVERSION, NATIVE_FORMATS


def test_native_formats_return_none(tmp_path):
    for ext in [".wav", ".mp3", ".flac"]:
        p = tmp_path / f"test{ext}"
        p.write_bytes(b"fake")
        assert convert_to_wav(p) is None


def test_unknown_ext_returns_none(tmp_path):
    p = tmp_path / "test.xyz"
    p.write_bytes(b"fake")
    assert convert_to_wav(p) is None


def test_needs_conversion_set():
    assert ".ogg" in NEEDS_CONVERSION
    assert ".m4a" in NEEDS_CONVERSION
    assert ".aac" in NEEDS_CONVERSION


def test_native_formats_set():
    assert ".wav" in NATIVE_FORMATS
    assert ".mp3" in NATIVE_FORMATS
    assert ".flac" in NATIVE_FORMATS
