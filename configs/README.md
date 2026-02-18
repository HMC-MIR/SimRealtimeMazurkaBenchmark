# Configuration Files

This directory contains JSON configuration files for the benchmark pipeline.

## Files

### `default_systems.json`
Default configurations for all supported systems:
- **DTW**: Dynamic Time Warping
- **NOA**: Normalized Online Alignment
- **NOA_MONOTONIC**: Normalized Online Alignment with monotonic constraint
- **MATCH**: MATCH algorithm (requires Java)
- **OLTW**: Online Time Warping (using Java implementation)
- **OLTW_GLOBAL**: Global OLTW (custom Python implementation)

### `oltw_global_examples.json`
Example OLTW_GLOBAL configurations with different parameter settings (A-F) as described in notebook 02.

## Usage

### Using Default Configurations
If no config file is specified, the script will use default configurations:

```bash
python benchmark.py run --benchmark train_small --systems DTW NOA
```

### Using Pre-defined Configurations
You can use one of the provided config files:

```bash
python benchmark.py experiment --benchmark train_small --config configs/default_systems.json
```

### Using OLTW_GLOBAL Presets
To run specific OLTW_GLOBAL settings:

```bash
python benchmark.py experiment --benchmark train_small \
  --systems OLTW_GLOBAL_A OLTW_GLOBAL_B \
  --config configs/oltw_global_examples.json
```

### Creating Custom Configurations
Create your own JSON file with system configurations. Example:

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

Then run:

```bash
python benchmark.py experiment --benchmark train_small \
  --systems MY_CUSTOM_SYSTEM \
  --config my_custom_config.json
```

## Parameter Descriptions

### Common Parameters
- `feat_dir`: Directory containing precomputed features
- `sr`: Sample rate (default: 22050 Hz)
- `hop_length`: Hop length for feature extraction (default: 512 samples)
- `distance_metric`: Distance metric ("cosine" or "euclidean")

### DTW/NOA Parameters
- `steps`: DTW step pattern as 2D array [[x1,y1], [x2,y2], ...]
- `weights`: Step weights

### OLTW_GLOBAL Parameters
- `c`: Constraint window (null for global alignment)
- `DTW_steps`: DTW step pattern for alignment
- `DTW_weights`: DTW step weights
- `window_steps`: Step pattern for window progression
