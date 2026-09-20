"""
Check that Vienna 4x22 match-file times are audio times.

The ground truth this benchmark derives from the match files is only as good as
the assumption that a performed note's time in the match file is its time in the
.wav. This script tests that assumption directly: it detects onsets in the
recording and measures what fraction of them land within a tolerance of a note
onset in the match file, scanning over candidate shifts. A correctly aligned
recording scores near 1.0 at a shift of zero.

Measuring agreement on every onset, rather than correlating onset envelopes,
matters here. An envelope correlation returns a confident-looking peak near zero
for any pair of signals with similar overall density, so it cannot distinguish
"aligned" from "no information"; the hit rate falls off sharply when a recording
is genuinely misaligned, and says so.

Two red herrings worth knowing about, both settled by this test:

  - Every Chopin op. 10 no. 3 and op. 38 performance has its first note at tick
    0, unlike the Mozart and Schubert ones. That is not a missing time shift:
    the Chopin recordings were trimmed to begin at the first note.
  - The Chopin <piece>_FirstOnsets.txt files shipped with the audio date from
    2001 and describe the untrimmed recordings, so they disagree with both the
    match files and the current audio. The Mozart and Schubert ones were
    annotated in 2016 against the current audio and do agree.

Usage:
    python -m scripts.vienna4x22.verify_timebase --audio-dir <dir> --match-dir <dir>
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import librosa as lb
import numpy as np

from scripts.vienna4x22.match_io import load_match

SR = 22050
HOP = 256
MAX_SHIFT_SEC = 4.0
SHIFT_STEP_SEC = 0.01
TOLERANCE_SEC = 0.05
# Below this, the match times do not describe the recording and the derived
# annotations would be wrong by however far off they are.
MIN_HIT_RATE = 0.80


def detect_onsets(path: Path) -> np.ndarray:
    """Onset times in a recording, in seconds."""
    y, sr = lb.load(str(path), sr=SR, mono=True)
    return lb.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units='time', backtrack=True)


def hit_rate(audio_onsets: np.ndarray, match_onsets: np.ndarray, shift: float,
             tolerance: float = TOLERANCE_SEC) -> float:
    """Fraction of detected audio onsets within tolerance of a shifted match onset."""
    if len(audio_onsets) == 0 or len(match_onsets) == 0:
        return 0.0
    shifted = match_onsets + shift
    idx = np.clip(np.searchsorted(shifted, audio_onsets), 1, len(shifted) - 1)
    nearest = np.minimum(np.abs(audio_onsets - shifted[idx - 1]),
                         np.abs(audio_onsets - shifted[idx]))
    return float(np.mean(nearest < tolerance))


def best_shift(audio_onsets: np.ndarray, match_onsets: np.ndarray) -> tuple:
    """
    Scan candidate shifts and return the one that explains the most onsets.

    Returns:
        (shift_seconds, hit_rate_at_that_shift, hit_rate_at_zero)
    """
    shifts = np.arange(-MAX_SHIFT_SEC, MAX_SHIFT_SEC + SHIFT_STEP_SEC, SHIFT_STEP_SEC)
    rates = np.array([hit_rate(audio_onsets, match_onsets, s) for s in shifts])
    peak = int(np.argmax(rates))
    return float(shifts[peak]), float(rates[peak]), hit_rate(audio_onsets, match_onsets, 0.0)


def segment_shifts(audio_onsets: np.ndarray, match_onsets: np.ndarray, duration: float) -> tuple:
    """
    Best shift over the first and last third of a recording.

    A constant offset moves both equally. A clock-rate mismatch between the MIDI
    capture and the audio recorder would instead show as a shift that grows
    across the piece, which no single offset could correct.
    """
    third = duration / 3.0
    head = audio_onsets[audio_onsets <= third]
    tail = audio_onsets[audio_onsets >= 2 * third]
    if len(head) < 10 or len(tail) < 10:
        return float('nan'), float('nan')
    return best_shift(head, match_onsets)[0], best_shift(tail, match_onsets)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--audio-dir', required=True, type=Path, help='unpacked Vienna 4x22 audio')
    parser.add_argument('--match-dir', required=True, type=Path, help='the corpus match/ directory')
    parser.add_argument('--emit-offsets', type=Path,
                        help='write any needed corrections as JSON, for prepare.py --offsets')
    args = parser.parse_args()

    audio_files = {p.stem: p for p in args.audio_dir.rglob('*.wav')}

    by_piece = defaultdict(list)
    for match_file in sorted(args.match_dir.glob('*.match')):
        name = re.match(r'(.+)_(p\d\d)$', match_file.stem)
        if not name:
            print(f"  skipping {match_file.name}: not a <piece>_p<NN> match file")
            continue
        by_piece[name.group(1)].append((name.group(2), match_file))

    corrections = {}
    checked = misaligned = 0

    for piece, entries in sorted(by_piece.items()):
        rows = []
        for performer, match_file in entries:
            if match_file.stem not in audio_files:
                print(f"  {piece} {performer}: no audio ({match_file.stem}.wav)")
                continue

            performance = load_match(match_file)
            match_onsets = np.array(sorted(t for ts in performance.onsets.values() for t in ts))
            audio_onsets = detect_onsets(audio_files[match_file.stem])
            duration = lb.get_duration(path=str(audio_files[match_file.stem]))

            shift, rate_at_shift, rate_at_zero = best_shift(audio_onsets, match_onsets)
            head, tail = segment_shifts(audio_onsets, match_onsets, duration)
            rows.append((performer, shift, rate_at_shift, rate_at_zero, tail - head))

            checked += 1
            if rate_at_zero < MIN_HIT_RATE:
                misaligned += 1
                corrections[match_file.stem] = round(shift, 4)

        if not rows:
            continue
        at_zero = np.array([r[3] for r in rows])
        shifts = np.array([r[1] for r in rows])
        drift = np.array([r[4] for r in rows])
        verdict = 'ALIGNED' if np.all(at_zero >= MIN_HIT_RATE) else 'MISALIGNED'
        print(f"{piece:24s} n={len(rows):2d}  onsets explained at shift 0: "
              f"{at_zero.mean():5.1%} (worst {at_zero.min():5.1%})  "
              f"best shift |max| {np.abs(shifts).max():.2f}s  "
              f"drift |max| {np.nanmax(np.abs(drift)):.2f}s  {verdict}")
        for performer, shift, rate_at_shift, rate_at_zero, d in rows:
            if rate_at_zero < MIN_HIT_RATE:
                print(f"      {performer}: only {rate_at_zero:.1%} at shift 0; "
                      f"best {rate_at_shift:.1%} at {shift:+.3f}s")

    print(f"\n{checked} recordings checked, {misaligned} misaligned")
    if args.emit_offsets and corrections:
        args.emit_offsets.write_text(json.dumps(corrections, indent=2, sort_keys=True) + '\n')
        print(f"wrote {len(corrections)} corrections to {args.emit_offsets}")
    elif misaligned == 0:
        print("No correction needed; prepare.py can be run without --offsets.")


if __name__ == '__main__':
    main()
