import io
import struct
import math
import wave

import pytest
from httpx import ASGITransport, AsyncClient

from transcription.main import app


def _make_wav_bytes(frequency: float = 440.0, duration: float = 1.0) -> bytes:
    """Create a WAV file in memory and return its bytes."""
    sample_rate = 22050
    amplitude = 16000
    n_samples = int(sample_rate * duration)

    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = []
        for i in range(n_samples):
            t = i / sample_rate
            value = int(amplitude * math.sin(2 * math.pi * frequency * t))
            samples.append(struct.pack("<h", value))
        wf.writeframes(b"".join(samples))

    return buf.getvalue()


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_index():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "audio" in resp.text.lower() or "midi" in resp.text.lower()


@pytest.mark.asyncio
async def test_transcribe_wav():
    wav_bytes = _make_wav_bytes()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/transcribe",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
    # 200 = success, 422 = zero notes (both are valid for a simple sine)
    assert resp.status_code in (200, 422)
    if resp.status_code == 200:
        assert resp.headers["content-type"] == "audio/midi"
        assert len(resp.content) > 0


@pytest.mark.asyncio
async def test_reject_unsupported_format():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/transcribe",
            files={"file": ("test.txt", b"not audio", "text/plain")},
        )
    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_reject_fake_wav():
    """A .wav file with wrong content should be rejected."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/transcribe",
            files={"file": ("fake.wav", b"this is not a wav file at all!!" * 10, "audio/wav")},
        )
    assert resp.status_code == 400
    assert "wav" in resp.json()["detail"].lower() or "形式" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_reject_tiny_file():
    """A file that's too small to be valid audio."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/transcribe",
            files={"file": ("tiny.wav", b"RIFF" + b"\x00" * 10, "audio/wav")},
        )
    assert resp.status_code == 400
    assert "小さすぎ" in resp.json()["detail"]
