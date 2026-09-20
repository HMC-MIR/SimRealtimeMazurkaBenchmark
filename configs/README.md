# Configuration Files

JSON files passed to `benchmark.py --config`. Each top-level key is a system name; its
value is the keyword arguments handed to that system's runner. A key becomes its own
experiment directory, so the same algorithm can be run several times under different
names.

## Files

| file | contents |
|---|---|
| `default_systems.json` | the parameters used for the reported results |
| `oltw_global_examples.json` | OLTW-Global step-pattern sweep, settings A-F (`c: null`) |
| `oltw_examples.json` | the same sweep for the windowed variant (`c: 500`) |
| `weight_sweep_examples.json` | DTW/SOA step-weight sweep used during tuning |

If `--config` is omitted, `get_default_configs()` in `benchmark.py` supplies equivalent
defaults, so the common case needs no config file at all:

```bash
python benchmark.py run --benchmark train_small --systems DTW SOA
```

## Usage

```bash
# reported configuration
python benchmark.py experiment --benchmark test --config configs/default_systems.json \
  --systems DTW SOA SOA_MONOTONIC OLTW_GLOBAL OLTW_OURS

# a parameter sweep: each key lands in its own experiment directory
python benchmark.py experiment --benchmark train_small \
  --systems OLTW_GLOBAL_A OLTW_GLOBAL_B \
  --config configs/oltw_global_examples.json
```

Custom configurations are just another JSON file:

```json
{
  "MY_CUSTOM_SYSTEM": {
    "hop_length": 512,
    "feat_dir": "features/chroma_stft_norm2",
    "sr": 22050,
    "distance_metric": "cosine",
    "c": null,
    "DTW_steps": [[1, 0], [0, 1], [1, 1]],
    "DTW_weights": [1, 1, 1],
    "window_steps": [[1, 1], [1, 0], [0, 1]]
  }
}
```

The runner picks the algorithm from the *name*, not the config: keys starting with `DTW`
run offline DTW, `SOA` run our online alignment, `MM_` run a MatchMaker baseline, and anything else
containing `OLTW_` runs our OLTW. See `ExperimentRunner.run` in `utils/experiments.py`.
An optional `_description` field is ignored by the runner and is there to document a setting.

## Parameters

### Common
- `feat_dir`: directory of precomputed features
- `sr`: sample rate (default 22050)
- `hop_length`: hop length in samples (default 512)
- `distance_metric`: `cosine` or `euclidean`

### DTW / SOA (`DTW*`, `SOA*`)
- `steps`: step pattern, `[[x1, y1], [x2, y2], ...]`
- `weights`: one weight per step
- `norm`: normalize the accumulated cost matrix (SOA only)
- `monotonic`: force predictions never to move backwards (SOA only)

### OLTW (`OLTW_GLOBAL*`, `OLTW_OURS*`)
- `c`: constraint window in frames, or `null` for an unconstrained search
- `DTW_steps`, `DTW_weights`: step pattern used inside the alignment
- `window_steps`: step pattern governing how the search window advances

### MatchMaker (`MM_*`)
- `method`: `dixon` or `arzt`
- `window_size`: search window in seconds
- `step_size`: max reference frames advanced per query frame (`arzt` only)
- `readout`: `reduced` or `raw` (`dixon` only); see `matchmaker_worker.READOUTS`

These run in a separate conda env because `pymatchmaker` pins `numpy<2`. The interpreter is
found via `$MATCHMAKER_PYTHON`, or as a sibling env named `matchmaker` of the active one.
See the main README.

### MATCH (`MATCH`)
- `audio_root`: directory holding the source recordings

Runs Dixon's MATCH through `sonic-annotator` and the match-vamp-plugin, at the sample rate
and hop length from the original paper (44100 / 882) with Euclidean distance.
