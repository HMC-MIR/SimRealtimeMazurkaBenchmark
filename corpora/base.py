"""
The corpus abstraction that the benchmark pipeline resolves paths through.

A corpus is a directory tree of recordings plus matching beat annotations:

    <audio_root>/<piece_id>.wav
    <annot_root>/<piece_id><annot_ext>

Everything downstream of scenario generation (the systems, eval_tools, the
notebooks) sees only the per-scenario symlinks, so a corpus is the single place
that knows where a benchmark's data actually lives.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Corpus:
    """
    Where one corpus keeps its audio and annotations, and how its piece IDs are
    spelled.

    A piece ID is normally qualified by its piece directory, e.g.
    "Chopin_Op024No2/Chopin_Op024No2_Ax-1985_pid9179-07". The train benchmarks
    predate that convention and store bare recording names in cfg/*.pkl; those
    benchmarks set default_piece_root so the missing directory is filled in
    here rather than guessed from the shape of the string.
    """

    name: str
    audio_root: str
    annot_root: str
    annot_ext: str = ".beat"
    default_piece_root: Optional[str] = None
    # Qualified piece IDs left out of every benchmark built from this corpus.
    excluded: FrozenSet[str] = frozenset()

    def qualify(self, piece_id: str) -> str:
        """Expand a possibly-bare piece ID to one relative to the corpus roots."""
        if self.default_piece_root:
            return f"{self.default_piece_root}/{piece_id}"
        return piece_id

    def audio_path(self, piece_id: str) -> str:
        return f"{self.audio_root}/{self.qualify(piece_id)}.wav"

    def annot_path(self, piece_id: str) -> str:
        return f"{self.annot_root}/{self.qualify(piece_id)}{self.annot_ext}"

    @property
    def id_audio_root(self) -> str:
        """
        The base directory that piece IDs resolve against, i.e. the root for
        which f"{id_audio_root}/{piece_id}.wav" == audio_path(piece_id).

        MATCH is the one system that takes audio paths rather than cached
        features, so it needs this rather than the corpus root itself.
        """
        if self.default_piece_root:
            return f"{self.audio_root}/{self.default_piece_root}"
        return self.audio_root


def build_dataset(
    corpus: Corpus,
    piece_roots: Sequence[str],
    logger: logging.Logger,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Enumerate a corpus's recordings from disk and build within-piece pairs.

    Recordings are discovered rather than read from cfg/, so adding or removing
    a recording changes the benchmark without editing a checked-in list. Pairs
    are within-piece and one-directional (i < j); aligning A to B and B to A
    would double the work without adding a distinct scenario.

    Args:
        corpus: the corpus to enumerate
        piece_roots: piece directories under the corpus audio root
        logger: logger instance

    Returns:
        piece_ids: qualified IDs, e.g. Chopin_Op024No2/<recording>
        pairs_list: list of (query, reference)
    """
    piece_ids: List[str] = []
    pairs_list: List[Tuple[str, str]] = []

    for piece_root in piece_roots:
        piece_audio_dir = Path(corpus.audio_root) / piece_root
        if not piece_audio_dir.exists():
            logger.warning(f"Audio directory not found: {piece_audio_dir}")
            continue

        local_piece_ids = sorted(f"{piece_root}/{p.stem}" for p in piece_audio_dir.glob("*.wav"))
        for piece_id in sorted(set(local_piece_ids) & corpus.excluded):
            logger.info(f"Excluding {piece_id} (listed in {corpus.name}'s excluded recordings)")
        local_piece_ids = [p for p in local_piece_ids if p not in corpus.excluded]
        if len(local_piece_ids) < 2:
            logger.warning(f"Found fewer than 2 recordings in {piece_audio_dir}; no pairs will be created")

        piece_ids.extend(local_piece_ids)
        for i in range(len(local_piece_ids)):
            for j in range(i + 1, len(local_piece_ids)):
                pairs_list.append((local_piece_ids[i], local_piece_ids[j]))

    logger.info(
        f"Built {corpus.name} dataset with {len(piece_ids)} recordings and {len(pairs_list)} pairs"
    )
    return piece_ids, pairs_list
