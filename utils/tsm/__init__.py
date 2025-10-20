"""
TSM utilities for piano concerto accompaniment.

This module provides tools for time-scale modification.
"""

# Import the main QueryGenerator class
from .tsm import TSM, online_tsm
from .conversion import to_tsm_path

__all__ = [
    'TSM',
    'online_tsm',
    'to_tsm_path'
]
