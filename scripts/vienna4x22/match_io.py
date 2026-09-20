"""
Reading Vienna 4x22 match files, and turning them into beat annotations.

A match file pairs each score note with the note the pianist actually played,
as one of three line forms:

    snote(<score note>)-note(<performed note>).   an aligned pair
    snote(<score note>)-deletion.                 in the score, not played
    insertion-note(<performed note>).             played, not in the score

Only the aligned pairs carry timing information usable as ground truth. The
performed onset is a MIDI tick; the file's info(midiClockUnits) and
info(midiClockRate) lines convert it to seconds.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Vienna 4x22 match files all declare 480 ticks per quarter at 500000 us per
# quarter, i.e. 960 ticks per second. Parsed per file rather than assumed.
_DEFAULT_CLOCK_UNITS = 480
_DEFAULT_CLOCK_RATE = 500000


@dataclass(frozen=True)
class Performance:
    """One performance's aligned notes, indexed by position in the score."""

    piece: str
    performer: str
    # score onset (in beats) -> performed onset times (in seconds), ascending
    onsets: Dict[float, List[float]]

    def score_onsets(self) -> List[float]:
        return sorted(self.onsets)


def _parse_info(line: str) -> Optional[Tuple[str, str]]:
    body = line[len("info(") : line.rindex(")")]
    key, _, value = body.partition(",")
    return key, value


def load_match(path: Path) -> Performance:
    """
    Read a match file's aligned notes.

    Args:
        path: path to a .match file

    Returns:
        a Performance holding every aligned note's score onset and performed time
    """
    clock_units, clock_rate = _DEFAULT_CLOCK_UNITS, _DEFAULT_CLOCK_RATE
    piece = performer = ""
    onsets: Dict[float, List[float]] = defaultdict(list)

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("info("):
                key, value = _parse_info(line)
                if key == "midiClockUnits":
                    clock_units = int(value)
                elif key == "midiClockRate":
                    clock_rate = int(value)
                elif key == "piece":
                    piece = value
                elif key == "performer":
                    performer = value
            elif line.startswith("snote(") and ")-note(" in line:
                score_part, perf_part = line.split(")-note(")
                # snote(id,[step,alter],octave,measure:beat,offset,duration,
                #       onset_beat,offset_beat,[voice,staff])
                # The trailing [voice,staff] holds commas, so cut it off first;
                # onset_beat is then the second-to-last field.
                fields = score_part[: score_part.rindex(",[")].split(",")
                onset_beat = float(fields[-2])
                # note(id,pitch,onset_tick,offset_tick,velocity,channel,track)
                onset_tick = int(perf_part.rstrip(").").split(",")[2])
                onsets[onset_beat].append(onset_tick)

    ticks_per_second = clock_units * 1_000_000 / clock_rate
    return Performance(
        piece=piece,
        performer=performer,
        onsets={beat: sorted(t / ticks_per_second for t in ticks) for beat, ticks in onsets.items()},
    )


def granularity_filter(onsets, integer_beats_only: bool) -> set:
    """
    Narrow a set of score positions to the annotation granularity.

    Annotating whole beats only weights the annotations evenly in time; keeping
    every score position instead weights dense passages more heavily.
    """
    if integer_beats_only:
        return {beat for beat in onsets if beat.is_integer()}
    return set(onsets)


def candidate_onsets(performances: Sequence[Performance], integer_beats_only: bool) -> List[float]:
    """
    Score positions that every performance has at least one note at.

    eval_tools pairs the query's and reference's annotations row by row, so every
    performance of a piece must be annotated at exactly the same positions. A
    position some pianist did not play cannot be timed there, so it is dropped
    for the whole piece.
    """
    shared = set.intersection(*(set(p.onsets) for p in performances))
    return sorted(granularity_filter(shared, integer_beats_only))


def drop_rolled(
    performances: Sequence[Performance],
    score_onsets: Sequence[float],
    max_spread: float,
) -> List[float]:
    """
    Drop positions whose notes are spread too far apart to time.

    The notes of one score onset are rarely simultaneous, and a rolled chord can
    span seconds: the final chord of the op. 38 Ballade spans over 5 s in some
    performances. No single timestamp represents such a position, so rather than
    pick one arbitrarily it is left unannotated.

    Args:
        performances: every performance of one piece
        score_onsets: positions to filter
        max_spread: largest tolerated span, in seconds, in any performance

    Returns:
        the kept positions, ascending
    """
    return [
        beat
        for beat in score_onsets
        if all(p.onsets[beat][-1] - p.onsets[beat][0] <= max_spread for p in performances)
    ]


def common_score_onsets(
    performances: Sequence[Performance],
    max_spread: float,
    integer_beats_only: bool,
) -> List[float]:
    """Positions to annotate for one piece: shared by all performances, and timeable."""
    return drop_rolled(performances, candidate_onsets(performances, integer_beats_only), max_spread)


def beat_times(performance: Performance, score_onsets: Sequence[float]) -> np.ndarray:
    """
    Time each score onset in one performance.

    The notes of a score onset are rarely simultaneous: hands spread by tens of
    milliseconds, and chords are rolled. The median is used rather than the first
    note, so the timestamp tracks the middle of the chord instead of whichever
    hand happened to lead. At the positions kept by common_score_onsets the two
    differ by about 15 ms on average.

    Args:
        performance: the performance to time
        score_onsets: score positions to time, as chosen for the whole piece

    Returns:
        one timestamp per score onset, in seconds
    """
    return np.array([float(np.median(performance.onsets[beat])) for beat in score_onsets])


def write_beat_file(path: Path, times: np.ndarray, score_onsets: Sequence[float], header: Sequence[str]) -> None:
    """
    Write annotations in the benchmark's .beat format.

    eval_tools.read_start_times reads the first whitespace-separated field of
    each non-comment line as a time in seconds, and ignores '%' comments. The
    score position is written as a second field to keep the files readable and
    to make a misaligned pair of files obvious on inspection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in header:
            f.write(f"% {line}\n")
        f.write("% start_time[sec]\tscore_onset[beat]\n")
        for t, beat in zip(times, score_onsets):
            f.write(f"{t:.6f}\t{beat:g}\n")
