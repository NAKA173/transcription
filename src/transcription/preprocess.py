"""Audio preprocessing for improved transcription quality.

Applies normalization and noise reduction before passing audio
to Basic Pitch, improving accuracy especially for noisy or
poorly-recorded sources.
"""

import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def preprocess_audio(
    audio_path: Path,
    normalize: bool = True,
    noise_reduce: bool = True,
    target_sr: int = 22050,
) -> Path:
    """Preprocess audio file for better transcription.

    Args:
        audio_path: Path to input audio file.
        normalize: Apply peak normalization.
        noise_reduce: Apply spectral gating noise reduction.
        target_sr: Target sample rate.

    Returns:
        Path to preprocessed WAV file (temp file, caller should clean up).
    """
    y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)

    if normalize:
        y = _normalize(y)

    if noise_reduce:
        y = _reduce_noise(y, sr)

    # Write preprocessed audio to temp file
    out_path = Path(tempfile.mktemp(suffix=".wav", prefix="preproc_"))
    sf.write(str(out_path), y, sr)
    return out_path


def _normalize(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak normalization — scale so the loudest sample hits target_peak.

    This ensures Basic Pitch receives audio at a consistent level,
    regardless of the original recording volume.
    """
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y * (target_peak / peak)
    return y


def _reduce_noise(y: np.ndarray, sr: int) -> np.ndarray:
    """Spectral gating noise reduction.

    Estimates a noise profile from the quietest frames, then
    attenuates frequency bins that fall below the noise threshold.
    This preserves musical content while reducing background hiss/hum.
    """
    # Compute STFT
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimate noise profile from the quietest 10% of frames
    frame_energy = np.sum(magnitude ** 2, axis=0)
    n_noise_frames = max(1, int(len(frame_energy) * 0.1))
    noise_frame_indices = np.argsort(frame_energy)[:n_noise_frames]
    noise_profile = np.mean(magnitude[:, noise_frame_indices], axis=1, keepdims=True)

    # Only gate if there's a meaningful difference between noise and signal frames
    signal_frame_indices = np.argsort(frame_energy)[-n_noise_frames:]
    signal_profile = np.mean(magnitude[:, signal_frame_indices], axis=1, keepdims=True)
    snr = np.mean(signal_profile) / max(np.mean(noise_profile), 1e-10)

    # If SNR is very low (noise floor is nearly same as signal), skip gating
    if snr < 1.5:
        return y

    # Spectral gating: subtract noise floor, keep above zero
    gain_factor = 1.5  # how aggressively to gate
    threshold = noise_profile * gain_factor
    mask = np.maximum(magnitude - threshold, 0.0) / np.maximum(magnitude, 1e-10)

    # Smooth the mask to avoid artifacts
    from scipy.ndimage import uniform_filter
    mask = uniform_filter(mask, size=(3, 3))

    # Apply mask and reconstruct
    cleaned_stft = magnitude * mask * np.exp(1j * phase)
    y_clean = librosa.istft(cleaned_stft, hop_length=hop_length, length=len(y))

    return y_clean
