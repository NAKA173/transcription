"""Beat detection and note quantization using librosa."""

from pathlib import Path
from dataclasses import dataclass, field

import librosa
import numpy as np


@dataclass
class TimeSignature:
    numerator: int    # beats per measure (e.g. 3, 4, 6)
    denominator: int  # beat unit (e.g. 4 for quarter, 8 for eighth)
    confidence: float  # 0.0 - 1.0

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass
class BeatInfo:
    tempo: float
    beat_times: np.ndarray
    beat_interval: float  # average seconds per beat
    time_signature: TimeSignature | None = None

    @property
    def subdivisions(self) -> np.ndarray:
        """Generate sub-beat grid (16th notes) for finer quantization.

        Uses actual beat intervals (not fixed tempo) so the grid adapts
        to tempo changes within the track.
        """
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

    @property
    def tempo_changes(self) -> list[tuple[float, float]]:
        """Detect tempo changes from beat intervals.

        Returns list of (time, bpm) tuples for variable-tempo MIDI encoding.
        """
        if len(self.beat_times) < 3:
            return [(0.0, self.tempo)]

        changes = []
        prev_bpm = 0.0
        for i in range(len(self.beat_times) - 1):
            interval = self.beat_times[i + 1] - self.beat_times[i]
            if interval > 0:
                bpm = 60.0 / interval
                # Only record if BPM changed significantly (>5%)
                if abs(bpm - prev_bpm) / max(prev_bpm, 1) > 0.05:
                    changes.append((self.beat_times[i], bpm))
                    prev_bpm = bpm

        if not changes:
            changes = [(0.0, self.tempo)]

        return changes


def detect_beats(audio_path: Path) -> BeatInfo:
    """Detect tempo, beat positions, and time signature from audio.

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

    # Detect time signature
    time_sig = detect_time_signature(y, sr, beat_times)

    return BeatInfo(
        tempo=tempo_val,
        beat_times=beat_times,
        beat_interval=beat_interval,
        time_signature=time_sig,
    )


def detect_time_signature(
    y: np.ndarray, sr: int, beat_times: np.ndarray,
) -> TimeSignature:
    """Detect time signature by analyzing beat accent patterns.

    Approach:
    1. Compute onset strength at each beat position
    2. Look for periodic accent patterns (strong beats every N beats)
    3. Test groupings of 2, 3, 4, 6 to find the best fit
    """
    if len(beat_times) < 6:
        return TimeSignature(numerator=4, denominator=4, confidence=0.0)

    # Get onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.times_like(onset_env, sr=sr)

    # Sample onset strength at each beat position
    beat_strengths = np.zeros(len(beat_times))
    for i, bt in enumerate(beat_times):
        idx = np.argmin(np.abs(onset_times - bt))
        beat_strengths[i] = onset_env[idx]

    if np.max(beat_strengths) == 0:
        return TimeSignature(numerator=4, denominator=4, confidence=0.0)

    # Normalize
    beat_strengths = beat_strengths / np.max(beat_strengths)

    # Test different groupings: which one shows the strongest
    # periodic accent pattern?
    candidates = [
        (2, 4),  # 2/4
        (3, 4),  # 3/4
        (4, 4),  # 4/4
        (6, 8),  # 6/8
    ]

    best_score = -1.0
    best_sig = (4, 4)

    for num, denom in candidates:
        score = _score_grouping(beat_strengths, num)
        if score > best_score:
            best_score = score
            best_sig = (num, denom)

    # Confidence: how much better is the best vs. average
    scores = [_score_grouping(beat_strengths, n) for n, _ in candidates]
    avg_score = np.mean(scores)
    confidence = min(1.0, (best_score - avg_score) / max(avg_score, 0.01))
    confidence = max(0.0, confidence)

    return TimeSignature(
        numerator=best_sig[0],
        denominator=best_sig[1],
        confidence=round(confidence, 3),
    )


def _score_grouping(strengths: np.ndarray, group_size: int) -> float:
    """Score how well beats fit a grouping of `group_size`.

    A good grouping means beat 0 (downbeat) of each group is consistently
    stronger than the other beats in the group.
    """
    n = len(strengths)
    if n < group_size * 2:
        return 0.0

    # Average strength by position within the group
    position_strengths = np.zeros(group_size)
    position_counts = np.zeros(group_size)

    for i in range(n):
        pos = i % group_size
        position_strengths[pos] += strengths[i]
        position_counts[pos] += 1

    # Avoid division by zero
    position_counts = np.maximum(position_counts, 1)
    avg_by_position = position_strengths / position_counts

    if np.sum(avg_by_position) == 0:
        return 0.0

    # Score: ratio of downbeat strength to average of non-downbeat positions
    downbeat = avg_by_position[0]
    others = np.mean(avg_by_position[1:])

    if others == 0:
        return downbeat

    return downbeat / others


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
