"""Drum note classification — split drum MIDI into sub-parts.

Maps General MIDI drum notes to categories:
hi-hat, snare, kick, tom, cymbal/crash, ride.
"""

from dataclasses import dataclass

import pretty_midi

GM_DRUM_MAP: dict[int, str] = {
    35: "kick", 36: "kick",
    38: "snare", 40: "snare", 37: "snare",
    42: "hihat", 44: "hihat", 46: "hihat",
    41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
    49: "crash", 52: "crash", 55: "crash", 57: "crash",
    51: "ride", 53: "ride", 59: "ride",
}

DRUM_PART_NAMES: dict[str, str] = {
    "kick": "Kick (バスドラム)",
    "snare": "Snare (スネア)",
    "hihat": "Hi-Hat (ハイハット)",
    "tom": "Tom (タム)",
    "crash": "Crash (クラッシュシンバル)",
    "ride": "Ride (ライド)",
}

DRUM_PART_ORDER = ["kick", "snare", "hihat", "tom", "crash", "ride"]


def classify_drum_note(pitch: int) -> str:
    if pitch in GM_DRUM_MAP:
        return GM_DRUM_MAP[pitch]
    if pitch <= 36:
        return "kick"
    elif pitch <= 40:
        return "snare"
    elif pitch <= 46:
        return "hihat"
    elif pitch <= 50:
        return "tom"
    elif pitch <= 53:
        return "ride"
    else:
        return "crash"


@dataclass
class DrumSeparationResult:
    instruments: dict[str, pretty_midi.Instrument]
    note_counts: dict[str, int]


def separate_drum_notes(
    drum_instrument: pretty_midi.Instrument,
) -> DrumSeparationResult:
    parts: dict[str, list[pretty_midi.Note]] = {k: [] for k in DRUM_PART_ORDER}

    for note in drum_instrument.notes:
        category = classify_drum_note(note.pitch)
        parts[category].append(note)

    instruments = {}
    note_counts = {}

    for part_name in DRUM_PART_ORDER:
        notes = parts[part_name]
        if not notes:
            continue
        inst = pretty_midi.Instrument(
            program=0,
            is_drum=True,
            name=DRUM_PART_NAMES.get(part_name, part_name),
        )
        inst.notes = sorted(notes, key=lambda n: n.start)
        instruments[part_name] = inst
        note_counts[part_name] = len(notes)

    return DrumSeparationResult(instruments=instruments, note_counts=note_counts)
