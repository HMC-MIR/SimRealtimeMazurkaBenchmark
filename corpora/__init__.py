"""
Corpus registry.

Named `corpora` rather than `datasets` so that a local package cannot shadow
HuggingFace's `datasets`, which a future neural baseline would plausibly pull in.
"""

from dataclasses import replace
from typing import Optional

from corpora.base import Corpus, build_dataset
from corpora.mazurkas import MAZURKAS
from corpora.vienna4x22 import VIENNA4X22

REGISTRY = {
    MAZURKAS.name: MAZURKAS,
    VIENNA4X22.name: VIENNA4X22,
}

__all__ = ["Corpus", "build_dataset", "REGISTRY", "get"]


def get(name: str, default_piece_root: Optional[str] = None) -> Corpus:
    """
    Look up a corpus by name.

    Args:
        name: registry key, e.g. "mazurkas"
        default_piece_root: piece directory to qualify bare piece IDs with, for
            benchmarks whose checked-in ID lists predate the qualified form

    Returns:
        the Corpus, with default_piece_root applied if given
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown corpus {name!r}; known corpora: {', '.join(sorted(REGISTRY))}")
    corpus = REGISTRY[name]
    if default_piece_root:
        corpus = replace(corpus, default_piece_root=default_piece_root)
    return corpus
