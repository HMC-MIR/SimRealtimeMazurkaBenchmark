"""
Check that Vienna 4x22 match-file times line up with the audio recordings.

The ground truth this benchmark derives from the match files is only as good as
the assumption that a performed note's time in the match file is its time in the
.wav. The corpus maintainers shifted the match and MIDI files onto the audio
timeline in March 2024 (see CHANGES.md upstream), but the result is not uniform:
every one of the 22 Chopin op. 10 no. 3 and op. 38 performances starts at tick 0,
while the Mozart and Schubert performances start at plausibly varied times
between 0.14 s and 2.27 s. Twenty-two independent takes cannot all begin exactly
at t=0, which suggests the shift never landed on the two Chopin pieces, or was
clamped away by the `max(0, note_on + time_shift)` in the upstream script.

This script measures the offset directly. It builds an onset envelope from the
audio, builds a second one from the match file's note onsets, and reports the lag
that best aligns them. An offset near zero means the match times can be used as
they stand; a consistent non-zero offset means the annotations would be wrong by
that much for that recording.

Usage:
    python -m scripts.vienna4x22.verify_timebase --audio-dir <dir> --match-dir <dir>
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import librosa as lb
import numpy as np
from scipy import signal

from scripts.vienna4x22.match_io import load_match

SR = 22050
HOP = 512
MAX_LAG_SEC = 4.0


def match_onset_envelope(performance, n_frames: int) -> np.ndarray:
    """
    Build a frame-rate onset envelope from a match file's performed onsets.

    Each score position contributes one impulse per note played there, so dense
    chords weigh more than single notes, roughly as they do in an audio onset
    envelope.
    """
    env = np.zeros(n_frames)
    for times in performance.onsets.values():
        for t in times:
            frame = int(round(t * SR / HOP))
            if 0 <= frame < n_frames:
                env[frame] += 1.0
    return env


def best_lag_seconds(audio_env: np.ndarray, match_env: np.ndarray) -> tuple:
    """
    Find the lag that best aligns two envelopes.

    Zero-padded cross-correlation rather than a circular one: at these lags a
    circular shift would wrap several seconds of one envelope around to the
    other end, which is exactly the kind of contamination that would bias the
    measurement this script exists to make.

    Args:
        audio_env: onset strength computed from the recording
        match_env: onset impulses built from the match file

    Returns:
        (lag_seconds, sharpness). A positive lag means the audio lags the match
        file, i.e. match times need that much added. Sharpness is the peak
        correlation over the median absolute correlation across searched lags;
        a flat, ambiguous correlation scores near 1.
    """
    n = min(len(audio_env), len(match_env))
    a = audio_env[:n].astype(float)
    m = match_env[:n].astype(float)
    a = (a - a.mean()) / (a.std() + 1e-9)
    m = (m - m.mean()) / (m.std() + 1e-9)

    scores = signal.correlate(a, m, mode='full', method='fft')
    # Index k of a 'full' correlation corresponds to shifting m forward by
    # k - (n - 1) samples.
    all_lags = np.arange(len(scores)) - (n - 1)

    max_lag = int(round(MAX_LAG_SEC * SR / HOP))
    searched = np.abs(all_lags) <= max_lag
    scores, lags = scores[searched], all_lags[searched]

    peak = int(np.argmax(scores))
    sharpness = scores[peak] / (np.median(np.abs(scores)) + 1e-9)
    return lags[peak] * HOP / SR, sharpness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', required=True, type=Path, help='directory of Vienna 4x22 .wav files')
    parser.add_argument('--match-dir', required=True, type=Path, help='directory of .match files')
    parser.add_argument('--limit', type=int, default=0, help='check only the first N performances per piece')
    args = parser.parse_args()

    audio_files = {p.stem: p for p in args.audio_dir.rglob('*.wav')}

    by_piece = defaultdict(list)
    for match_file in sorted(args.match_dir.glob('*.match')):
        name = re.match(r'(.+)_(p\d\d)$', match_file.stem)
        if not name:
            print(f"  skipping {match_file.name}: not a <piece>_p<NN> match file")
            continue
        piece, performer = name.groups()
        by_piece[piece].append((performer, match_file))

    all_offsets = []
    for piece, entries in by_piece.items():
        if args.limit:
            entries = entries[:args.limit]
        offsets = []
        for performer, match_file in entries:
            if match_file.stem not in audio_files:
                print(f"  {piece} {performer}: NO AUDIO ({match_file.stem}.wav)")
                continue

            y, _ = lb.load(audio_files[match_file.stem], sr=SR, mono=True)
            audio_env = lb.onset.onset_strength(y=y, sr=SR, hop_length=HOP)

            performance = load_match(match_file)
            match_env = match_onset_envelope(performance, len(audio_env))

            lag, sharpness = best_lag_seconds(audio_env, match_env)
            offsets.append((performer, lag, sharpness))

        if not offsets:
            continue
        lags = np.array([o[1] for o in offsets])
        all_offsets.extend(lags)
        flag = 'OK' if np.all(np.abs(lags) <= 0.05) else 'OFFSET'
        print(f"{piece:24s} n={len(offsets):2d}  median={np.median(lags):+7.3f}s  "
              f"range=[{lags.min():+.3f}, {lags.max():+.3f}]s  "
              f"min sharpness={min(o[2] for o in offsets):5.1f}  {flag}")
        for performer, lag, sharpness in offsets:
            if abs(lag) > 0.05:
                print(f"      {performer}: {lag:+.3f}s (sharpness {sharpness:.1f})")

    if all_offsets:
        a = np.abs(np.array(all_offsets))
        print(f"\n{len(a)} performances checked; "
              f"{int((a > 0.05).sum())} with |offset| > 50 ms, "
              f"{int((a > 0.5).sum())} with |offset| > 500 ms")


if __name__ == '__main__':
    main()
