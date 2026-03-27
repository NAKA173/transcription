"""Beat detection and note quantization using librosa."""

from pathlib import Path
from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class BeatInfo:
    tempo: float
    beat_times: np.ndarray
    beat_interval: float  # seconds per beat

    @property
    def subdivisions(self) -> np.ndarray:
        """Generate sub-beat grid (16th notes) for finer quantization."""
        if len(self.beat_times) < 2:
            return self.beat_times
        subs = []
        for i in range(len(self.beat_times) - 1):
            start = self.beat_times[i]
            end = self.beat_times[i + 1]
            for j in range(4):  # 4 subdivisions per beat = 16th notes
                subs.append(start + (end - start) * j / 4)
        subs.append(self.beat_times[-1])
        return np.array(subs)


def detect_beats(audio_path: Path) -> BeatInfo:
    """Detect tempo and beat positions from audio using librosa.

    Uses onset-based beat tracking which is effective even on
    non-drum sources, but works best when drums are present.
    """
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)

    # Detect tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")

    # Convert frames to time positions
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # tempo can be an ndarray or scalar depending on librosa version
    tempo_val = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
    beat_interval = 60.0 / tempo_val if tempo_val > 0 else 0.5

    return BeatInfo(
        tempo=tempo_val,
        beat_times=beat_times,
        beat_interval=beat_interval,
    )


def quantize_note_times(
    start: float,
    end: float,
    grid: np.ndarray,
    strength: float = 1.0,
) -> tuple[float, float]:
    """Snap a note's start/end to the nearest grid position.

    Args:
        start: Note start time in seconds.
        end: Note end time in seconds.
        grid: Array of grid positions (beat or sub-beat times).
        strength: 0.0 = no quantization, 1.0 = full snap to grid.

    Returns:
        Tuple of (quantized_start, quantized_end).
    """
    if len(grid) == 0 or strength == 0.0:
        return start, end

    duration = end - start

    # Find nearest grid point for start
    idx = np.argmin(np.abs(grid - start))
    q_start = grid[idx]

    # Blend between original and quantized based on strength
    q_start = start + (q_start - start) * strength

    # Preserve note duration (don't quantize duration, just snap the start)
    q_end = q_start + duration

    # Ensure minimum duration
    if q_end <= q_start:
        q_end = q_start + 0.05

    return q_start, q_end
