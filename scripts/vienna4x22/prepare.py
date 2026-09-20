"""
Build the Vienna4x22/ corpus tree the benchmark reads.

The upstream corpus ships score-to-performance match files but no audio and no
beat annotations, so this script produces both halves of what corpora/vienna4x22.py
expects:

    Vienna4x22/wav_22050_mono/<piece>/<performer>.wav      transcoded audio
    Vienna4x22/annotations_beat/<piece>/<performer>.beat   derived annotations

Run it once, then use the benchmark normally:

    git clone https://github.com/CPJKU/vienna4x22
    curl -O https://repo.mdw.ac.at/projects/IWK/the_vienna_4x22_piano_corpus/data/audio.zip
    unzip audio.zip -d vienna4x22_audio
    python -m scripts.vienna4x22.prepare --audio-dir vienna4x22_audio --match-dir vienna4x22/match
    python benchmark.py run --benchmark vienna4x22 --systems DTW SOA --jobs 8

Audio with no corresponding match file (the averaged takes, the special versions)
is skipped, following the upstream setup_audio.py.

Before trusting the output, run scripts/vienna4x22/verify_timebase.py: the
annotations are only correct if match-file times are audio times. If that check
reports offsets, pass them back in with --offsets.
"""

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

import corpora
from scripts.vienna4x22.match_io import (
    beat_times,
    candidate_onsets,
    drop_rolled,
    granularity_filter,
    load_match,
    write_beat_file,
)
from utils.constants import DEFAULT_SR

# Score positions whose notes span more than this are rolled chords, with no one
# time that represents the position. See match_io.common_score_onsets.
DEFAULT_MAX_SPREAD = 0.200


def transcode(src: Path, dst: Path, sr: int) -> None:
    """Downmix and resample one recording, matching the Mazurka corpus layout."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src),
         '-ac', '1', '-ar', str(sr), str(dst)],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--audio-dir', required=True, type=Path, help='unpacked Vienna 4x22 audio')
    parser.add_argument('--match-dir', required=True, type=Path, help='the corpus match/ directory')
    parser.add_argument('--max-spread', type=float, default=DEFAULT_MAX_SPREAD,
                        help='drop score positions whose notes span more than this, in seconds')
    parser.add_argument('--all-score-onsets', action='store_true',
                        help='annotate every score position rather than whole beats only')
    parser.add_argument('--offsets', type=Path,
                        help='JSON mapping <piece>_<performer> to seconds to add to its match times')
    parser.add_argument('--skip-audio', action='store_true', help='write annotations only')
    args = parser.parse_args()

    corpus = corpora.get('vienna4x22')
    offsets = json.loads(args.offsets.read_text()) if args.offsets else {}

    audio_files = {p.stem: p for p in args.audio_dir.rglob('*.wav')}

    by_piece = defaultdict(dict)
    for match_file in sorted(args.match_dir.glob('*.match')):
        piece, performer = re.match(r'(.+)_(p\d\d)$', match_file.stem).groups()
        by_piece[piece][performer] = match_file

    total_annotated = total_transcoded = total_missing = 0

    for piece, entries in sorted(by_piece.items()):
        performances = {}
        for performer, match_file in entries.items():
            performance = load_match(match_file)
            shift = offsets.get(match_file.stem, 0.0)
            if shift:
                performance = type(performance)(
                    piece=performance.piece,
                    performer=performance.performer,
                    onsets={b: [t + shift for t in ts] for b, ts in performance.onsets.items()},
                )
            performances[performer] = performance

        # Chosen once per piece, not per recording: eval_tools pairs the
        # query's and reference's annotations row by row.
        entries_list = list(performances.values())
        candidates = candidate_onsets(entries_list, integer_beats_only=not args.all_score_onsets)
        score_onsets = drop_rolled(entries_list, candidates, args.max_spread)

        # Reported separately because they mean different things: a position
        # some pianist did not play, versus a rolled chord no one timestamp
        # represents.
        everywhere = granularity_filter(
            set().union(*(set(p.onsets) for p in entries_list)),
            integer_beats_only=not args.all_score_onsets,
        )
        unplayed = len(everywhere) - len(candidates)
        rolled = len(candidates) - len(score_onsets)

        for performer, performance in sorted(performances.items()):
            piece_id = f"{piece}/{performer}"
            stem = f"{piece}_{performer}"

            times = beat_times(performance, score_onsets)
            write_beat_file(
                Path(corpus.annot_path(piece_id)),
                times,
                score_onsets,
                header=[
                    f"beat annotation for {piece}, {performance.performer}",
                    "derived from the Vienna 4x22 match files by scripts/vienna4x22/prepare.py",
                    f"score positions common to all {len(performances)} performances, "
                    f"max within-position spread {args.max_spread:g}s"
                    + (f", match times shifted by {offsets[stem]:+.3f}s" if stem in offsets else ""),
                ],
            )
            total_annotated += 1

            if args.skip_audio:
                continue
            if stem not in audio_files:
                print(f"  no audio for {stem}")
                total_missing += 1
                continue
            dst = Path(corpus.audio_path(piece_id))
            if not dst.exists():
                transcode(audio_files[stem], dst, DEFAULT_SR)
                total_transcoded += 1

        spans = [beat_times(p, score_onsets)[-1] - beat_times(p, score_onsets)[0] for p in performances.values()]
        print(f"{piece:24s} {len(performances):2d} performances, {len(score_onsets):3d} annotated positions "
              f"({unplayed} not played everywhere, {rolled} rolled), {min(spans):.0f}-{max(spans):.0f}s")

    print(f"\nwrote {total_annotated} annotation files, transcoded {total_transcoded} recordings"
          + (f", {total_missing} missing audio" if total_missing else ""))
    print(f"annotations: {corpus.annot_root}/")
    if not args.skip_audio:
        print(f"audio:       {corpus.audio_root}/")


if __name__ == '__main__':
    main()
