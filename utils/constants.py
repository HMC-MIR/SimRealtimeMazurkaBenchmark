"""Constants shared across the benchmark."""

import numpy as np

# Audio / feature settings used by every system except MATCH, which follows
# Dixon's original configuration (sr=44100, hop_length=882).
DEFAULT_SR: int = 22050
DEFAULT_HOP_LENGTH: int = 512

DEFAULT_DTW_STEPS: np.ndarray = np.array([1, 1, 1, 2, 2, 1]).reshape((-1, 2))
DEFAULT_DTW_WEIGHTS: np.ndarray = np.array([1, 1, 2])

# Maps the system keys used on the command line and as experiment/eval
# directory names onto the labels used in the paper's tables and figures.
# Anything not listed here is displayed under its own key.
SYSTEM_DISPLAY_NAMES: dict = {
    "DTW": "DTW",
    "MATCH": "MATCH",
    "OLTW": "OLTW",
    "OLTW_GLOBAL": "OLTW-Global",
    "OLTW_OURS": "OLTW-Ours",
    "MM_DIXON": "MM-Dixon",
    "MM_ARZT": "MM-Arzt",
    "NOA": "SOA",
    "NOA_MONOTONIC": "SOA-Mono",
}


def display_name(system: str) -> str:
    """Return the paper label for a system key."""
    return SYSTEM_DISPLAY_NAMES.get(system, system)
