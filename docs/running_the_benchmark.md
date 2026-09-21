# Running the benchmark

A record of how the Vienna 4x22 results and the repaired Mazurka test results were
produced, with the commands, the timings they actually took, and the failures worth
knowing about before repeating this.

Everything here assumes the repository root as the working directory.

## The interpreter

Use the environment's interpreter by name:

```bash
PY=/home/ctang/ttmp/anaconda3/envs/online-alignment-dev/bin/python
```

`benchmark.py` imports `vamp` at module scope through `utils/experiments.py`, and that
package exists only in `online-alignment-dev`. Plain `python` resolves to conda `base`
in a fresh shell, so every command below fails at import with
`ModuleNotFoundError: No module named 'vamp'` unless the environment is activated.

This matters most for detached work. A shell started with `nohup ... &` does not inherit
an activated conda environment reliably across a session restart, and the failure is
instant and silent if nothing is watching the log. Naming the interpreter is the fix.

## Building the Vienna 4x22 corpus

The upstream repository ships match files but no audio and no beat annotations, so both
halves are produced locally before the benchmark can see the corpus.

```bash
git clone https://github.com/CPJKU/vienna4x22
curl -L -o audio.zip \
  https://repo.mdw.ac.at/projects/IWK/the_vienna_4x22_piano_corpus/data/audio.zip
unzip -q audio.zip -d vienna4x22_audio
```

The archive is 1.3 GB and unpacks to 1.6 GB across 112 `.wav` files at 44.1 kHz stereo,
laid out as `audio/<PieceDir>/<piece>_p<NN>.wav`. It carries more recordings than the 88
the benchmark uses: averaged takes and a `Chopin_Ballade_special-versions` directory have
no corresponding match file and are skipped.

Check the timebase before trusting anything derived from the match files:

```bash
$PY -m scripts.vienna4x22.verify_timebase \
    --audio-dir vienna4x22_audio --match-dir vienna4x22/match
```

This detects onsets in each recording and reports what fraction land within 50 ms of a
note onset in the match file, scanning candidate shifts. All 88 recordings currently
report 97.6% to 98.6% at a shift of zero, so no correction is needed. If a future
revision of the corpus breaks that, the script writes the corrections it measures with
`--emit-offsets`, and `prepare.py` takes them with `--offsets`.

Then build the corpus tree:

```bash
$PY -m scripts.vienna4x22.prepare \
    --audio-dir vienna4x22_audio --match-dir vienna4x22/match
```

This transcodes the 88 recordings to 22.05 kHz mono under `Vienna4x22/wav_22050_mono/`
(341 MB) and derives `Vienna4x22/annotations_beat/`, timing each score beat by the median
onset of the notes aligned to it. It prints one line per piece; the annotated position
counts should be 38, 176, 151 and 83.

## Running the systems

Scenario generation and feature caching take under a minute for this corpus:

```bash
$PY benchmark.py prepare  --benchmark vienna4x22     # 924 scenarios
$PY benchmark.py features --benchmark vienna4x22     # ~20 s
```

The systems were then run in groups, because their requirements differ. Measured
wall-clock for 924 scenarios each, on a 40-core machine:

```bash
# in-process, no external dependencies
$PY benchmark.py experiment --benchmark vienna4x22 \
    --systems DTW SOA SOA_MONOTONIC OLTW_OURS --jobs 12
#   SOA            2m23s      (--jobs 8)
#   DTW            8m34s      (--jobs 8)
#   SOA_MONOTONIC  <1s        derived from the SOA paths, not recomputed
#   OLTW_OURS      2m10s

# Dixon's OLTW, forks a JVM per scenario; needs match/PerformanceMatcher.jar
$PY benchmark.py experiment --benchmark vienna4x22 --systems OLTW --jobs 8
#   OLTW          12m33s

# MatchMaker runs out of process in its own environment
export MATCHMAKER_PYTHON=/home/ctang/ttmp/anaconda3/envs/matchmaker/bin/python
$PY benchmark.py experiment --benchmark vienna4x22 --systems MM_DIXON MM_ARZT --jobs 8
#   MM_DIXON      87m10s     by far the slowest
#   MM_ARZT        4m57s     bounded by step_size=3, so far cheaper than MM_DIXON

$PY benchmark.py experiment --benchmark vienna4x22 --systems OLTW_GLOBAL --jobs 12
#   OLTW_GLOBAL    2m03s
```

Roughly two hours end to end, about three quarters of it `MM_DIXON`. Pass `--resume` when
re-running so finished systems are kept rather than cleared.

Then evaluate and read the results:

```bash
$PY benchmark.py evaluate --benchmark vienna4x22
$PY -m scripts.compare_benchmarks --benchmarks test vienna4x22 --out results/cross_corpus.csv
```

`compare_benchmarks` prints both corpora side by side and flags any row whose coverage
falls short of its benchmark. Read that warning block: a system can appear in every
scenario and still be scored on a small subset.

## Plotting

`03_Evaluate.ipynb` holds `plotErrorVsTolerance`. Point it at the Vienna eval directory
and save under a distinct name so the Mazurka figure is not overwritten:

```python
EVAL_DIR = "eval_vienna4x22"
systems = ["DTW", "OLTW", "MM_ARZT", "MM_DIXON",
           "OLTW_OURS", "OLTW_GLOBAL", "SOA", "SOA_MONOTONIC"]
tols = [50, 100, 200, 500, 1000, 2000]
df = plotErrorVsTolerance(EVAL_DIR, systems, tols,
                          savefile="figures/error_rate_vs_tolerance_vienna4x22.png")
```

`style='line'` reads better on this corpus than the default bars: the systems fall into a
narrower band than on the Mazurkas, and eight adjacent bars per tolerance are hard to
separate.

## Failures worth knowing about

### DTW segfaults when a pair is too unequal in length

`DEFAULT_DTW_STEPS = [[1,1],[1,2],[2,1]]` caps the warping path's slope at 2, so the query
can advance at most two frames per reference frame. When a pair's duration ratio exceeds
that, no valid path exists, and `hmc_mir.align.dtw` does not raise: it allocates without
bound (observed at 178 GB on a 187 GB machine) and then segfaults.

Two consequences. A segfault is not a Python exception, so the `try/except` in
`ExperimentRunner.run_batch` cannot catch it. And under `multiprocessing.Pool`, the worker
dies, `Pool` silently replaces it, and `imap_unordered` waits forever for a result that
will never arrive: every worker sits idle at 0% CPU while the job appears to be running.
If a run stalls with no progress, sample `/proc/<pid>/stat` rather than trusting `ps`,
whose `%CPU` is a lifetime average and will still show the last busy figure.

On the Mazurka test set this affects 7 of 3802 scenarios, all pairing
`Chopin_Op068No3_Cortot-1951_pid9066b-19` (164.7 s, against a median of 98 s for that
piece, because it takes repeats the others do not) with the shortest performances. Their
ratios are 2.035 to 2.296; every scenario below 2.0 succeeds. That recording is not listed
in `Chopin_Mazurkas/annotations_beat/structure_exceptions.txt`, and arguably should be.

No Vienna pair comes close: the worst ratio is 1.740, because the four excerpts are short
and all 22 pianists play the same written material, with no repeat-structure divergence.

### Chaining jobs

When waiting for one detached job before starting the next, wait on a sentinel the first
job writes into its log:

```bash
while ! grep -q "ALL SYSTEMS DONE" "$LOG"; do sleep 30; done
```

Do not wait with `pgrep -f <pattern>`. The pattern appears in the waiting script's own
command line, so `pgrep` matches itself and the loop never exits. That cost about 25
minutes of a silent no-op before it was noticed.
