"""Constants shared across the benchmark."""

import numpy as np

# Audio / feature settings used by every system except MATCH, which follows
# Dixon's original configuration (sr=44100, hop_length=882).
DEFAULT_SR: int = 22050
DEFAULT_HOP_LENGTH: int = 512

DEFAULT_DTW_STEPS: np.ndarray = np.array([1, 1, 1, 2, 2, 1]).reshape((-1, 2))
DEFAULT_DTW_WEIGHTS: np.ndarray = np.array([1, 1, 2])

# System keys double as command-line names and as experiment/eval directory
# names. Most read the same in the paper; the few that do not are listed here,
# and anything absent is displayed under its own key.
SYSTEM_DISPLAY_NAMES: dict = {
    "SOA_MONOTONIC": "SOA-Mono",
    "OLTW_GLOBAL": "OLTW-Global",
    "OLTW_OURS": "OLTW-Ours",
    "MM_DIXON": "MM-Dixon",
    "MM_ARZT": "MM-Arzt",
}


def display_name(system: str) -> str:
    """Return the paper label for a system key."""
    return SYSTEM_DISPLAY_NAMES.get(system, system)
