"""
Build one error-rate table spanning several benchmarks.

03_Evaluate.ipynb reports a single benchmark. Putting two corpora beside each
other is what shows the effect this benchmark set is designed to isolate: the
Mazurka scenarios pair recordings made decades apart on different pianos, while
every Vienna 4x22 performance of a piece shares an instrument, a room and a
microphone setup. The gap between the two columns for one system is the cost of
recording-condition variation, holding the aligner fixed.

Usage:
    python -m scripts.compare_benchmarks --benchmarks test vienna4x22
    python -m scripts.compare_benchmarks --benchmarks test vienna4x22 --out results/cross_corpus.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from corpora.benchmarks import BENCHMARK_CONFIGS
from utils.constants import display_name

DEFAULT_TOLERANCES = [50, 100, 200, 500, 1000, 2000]


def load_errors(path: Path) -> np.ndarray:
    """
    Read one errs.pkl as a flat array of signed errors in seconds.

    Coerced through float rather than concatenated directly: eval runs predating
    the current eval_tools left None in place of an unscored scenario, which
    makes an object array that np.isnan cannot handle. Those become NaN here and
    are dropped with the rest, so a partial old result reports on what it
    actually scored instead of raising.
    """
    with open(path, 'rb') as f:
        per_scenario = pickle.load(f)
    arrays = [
        np.asarray(v, dtype=float).ravel()
        for v in per_scenario.values()
        if v is not None and np.asarray(v).size
    ]
    if not arrays:
        return np.array([])
    errors = np.concatenate(arrays)
    return errors[~np.isnan(errors)]


def coverage(eval_dir: Path, system: str, scenarios_dir: Path) -> tuple:
    """How many scenarios were scored, out of how many exist."""
    with open(eval_dir / system / 'errs.pkl', 'rb') as f:
        scored = len(pickle.load(f))
    total = sum(1 for p in scenarios_dir.iterdir() if p.is_dir()) if scenarios_dir.exists() else 0
    return scored, total


def warn_partial_coverage(table: pd.DataFrame) -> None:
    """
    Flag rows that do not cover their whole benchmark.

    Two ways a row can fall short, and the second is the one that bites. A
    system can be missing scenarios outright, which the scenario count shows.
    Or it can appear in every scenario while contributing almost no scored
    annotations, which happens when an unfinished experiment leaves entries an
    older eval_tools wrote as None. The error rate is then computed over a small
    and arbitrary subset while the row looks complete, so annotation counts are
    compared against the best-covered system in the same benchmark.
    """
    problems = []
    for (benchmark, system), row in table.iterrows():
        scored, _, total = str(row['scenarios']).partition('/')
        if total and scored != total:
            problems.append(f"{benchmark}/{system}: {scored} of {total} scenarios scored")

        best = table.loc[benchmark, 'annotations'].max()
        if row['annotations'] < 0.9 * best:
            problems.append(
                f"{benchmark}/{system}: {row['annotations']:,} annotations vs "
                f"{best:,} for the best-covered system in this benchmark "
                f"({row['annotations'] / best:.0%})"
            )

    if problems:
        print("\nIncomplete rows -- these error rates are not computed over the whole benchmark:")
        for problem in problems:
            print(f"  {problem}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--benchmarks', nargs='+', required=True, choices=list(BENCHMARK_CONFIGS))
    parser.add_argument('--tolerances', nargs='+', type=int, default=DEFAULT_TOLERANCES,
                        help='error tolerances in milliseconds')
    parser.add_argument('--out', type=Path, help='write the table as CSV')
    args = parser.parse_args()

    records = []
    for benchmark in args.benchmarks:
        config = BENCHMARK_CONFIGS[benchmark]
        eval_dir = Path(config['eval_dir'])
        scenarios_dir = Path(config['scenarios_dir'])
        if not eval_dir.exists():
            print(f"{benchmark}: no eval directory at {eval_dir}, skipping")
            continue

        for system_dir in sorted(p for p in eval_dir.iterdir() if (p / 'errs.pkl').is_file()):
            system = system_dir.name
            errors = load_errors(system_dir / 'errs.pkl')
            if errors.size == 0:
                print(f"{benchmark}/{system}: no scored errors, skipping")
                continue
            scored, total = coverage(eval_dir, system, scenarios_dir)
            row = {
                'benchmark': benchmark,
                'system': display_name(system),
                'scenarios': f"{scored}/{total}" if total else str(scored),
                'annotations': errors.size,
            }
            row.update({f'>{t}ms': 100 * np.mean(np.abs(errors) > t / 1000) for t in args.tolerances})
            records.append(row)

    if not records:
        print("nothing to report")
        return

    table = pd.DataFrame(records).set_index(['benchmark', 'system'])
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(table.round(2))

    warn_partial_coverage(table)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out)
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
