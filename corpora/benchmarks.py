"""
The benchmarks the pipeline can run, and the corpora they draw from.

Kept apart from benchmark.py so that the evaluation notebooks can look up a
benchmark's directories without importing the pipeline, and with it librosa and
numba.
"""

from typing import Any, Dict

import corpora
from corpora import mazurkas, vienna4x22

# Each benchmark names the corpus it draws from, where its scenario/experiment/
# eval directories live, and how its recording list is obtained: benchmarks with
# 'piece_roots' enumerate those directories from disk, while the older train
# benchmarks read a checked-in list from cfg/. 'default_piece_root' is set for
# benchmarks whose cfg/ lists hold bare recording names (see corpora.base.Corpus).
BENCHMARK_CONFIGS = {
    'train_small': {
        'corpus': 'mazurkas',
        'default_piece_root': mazurkas.TRAIN_PIECE_ROOT,
        'train_file': 'cfg/mazurkas.train.pkl',
        'pair_file': 'cfg/mazurkas.train_pairs.pkl',
        'scenarios_dir': 'scenarios',
        'experiments_dir': 'experiments',
        'eval_dir': 'eval',
    },
    'train': {
        'corpus': 'mazurkas',
        'default_piece_root': mazurkas.TRAIN_PIECE_ROOT,
        'train_file': 'cfg/mazurkas.train_large.pkl',
        'pair_file': 'cfg/mazurkas.train_pairs_large.pkl',
        'scenarios_dir': 'scenarios_train',
        'experiments_dir': 'experiments_train',
        'eval_dir': 'eval_train',
    },
    'test': {
        'corpus': 'mazurkas',
        'piece_roots': mazurkas.TEST_PIECE_ROOTS,
        'train_file': 'cfg/mazurkas.test.pkl',
        'pair_file': 'cfg/mazurkas.test_pairs.pkl',
        'scenarios_dir': 'scenarios_test',
        'experiments_dir': 'experiments_test',
        'eval_dir': 'eval_test',
    },
    'vienna4x22': {
        'corpus': 'vienna4x22',
        'piece_roots': vienna4x22.PIECE_ROOTS,
        'train_file': 'cfg/vienna4x22.pkl',
        'pair_file': 'cfg/vienna4x22_pairs.pkl',
        'scenarios_dir': 'scenarios_vienna4x22',
        'experiments_dir': 'experiments_vienna4x22',
        'eval_dir': 'eval_vienna4x22',
    },
}


def get_corpus(config: Dict[str, Any]) -> corpora.Corpus:
    """Resolve the corpus a benchmark config refers to."""
    return corpora.get(config['corpus'], config.get('default_piece_root'))
