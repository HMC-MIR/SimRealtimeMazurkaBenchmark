#!/usr/bin/env python3
"""
Benchmark Pipeline Script

This script provides a command-line interface for running the SimRealtimeMazurkaBenchmark pipeline.
It supports preparing scenarios, computing features, running experiments, and evaluating results.
"""

import os
import sys
import argparse
import json
import pickle
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def _limit_worker_threads():
    """
    Caps the BLAS/numba thread pools to one thread per process when running with
    multiple workers, so that N worker processes do not each start a pool sized for
    the whole machine.

    Must run before numba is imported: numba snapshots the environment at import
    time, and changing NUMBA_NUM_THREADS afterwards makes every JIT compilation in a
    forked worker raise "Cannot set NUMBA_NUM_THREADS to a different value once the
    threads have been launched". argparse has not run yet, so --jobs is read straight
    off the command line.
    """
    jobs = 1
    if '--jobs' in sys.argv:
        try:
            jobs = int(sys.argv[sys.argv.index('--jobs') + 1])
        except (IndexError, ValueError):
            return
    if jobs > 1:
        for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
            os.environ.setdefault(var, '1')


_limit_worker_threads()

import numpy as np
import librosa as lb
from tqdm import tqdm

# Import local modules
import corpora
from corpora.benchmarks import BENCHMARK_CONFIGS, get_corpus
import utils.constants as constants
from utils.experiments import ExperimentRunner
from utils.match_features import extract_match_features
import eval_tools


# ============================================================================
# Configuration and Constants
# ============================================================================

# BENCHMARK_CONFIGS and get_corpus live in corpora.benchmarks, so that the
# evaluation notebooks can import them without pulling in librosa and numba.

FEAT_DIR = "features"


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"benchmark_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logging to {log_file}")
    return logger


# ============================================================================
# Data Preparation Functions
# ============================================================================

def validate_file(filepath: str, logger: logging.Logger) -> bool:
    """
    Validate that a file exists, is a regular file, and is not empty.
    
    Args:
        filepath: path to file
        logger: logger instance
        
    Returns:
        True if file is valid, False otherwise
    """
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return False
        
    if not os.path.isfile(filepath):
        logger.warning(f"Not a file: {filepath}")
        return False
        
    if os.path.getsize(filepath) == 0:
        logger.warning(f"File is empty: {filepath}")
        return False
        
    return True


def generate_scenarios(outdir: str, pairs_list: List[tuple], corpus: corpora.Corpus, logger: logging.Logger):
    """
    Generate all possible alignment scenarios for a given output directory.
    
    Args:
        outdir: output directory
        pairs_list: list of pairs of recordings to process. Each pair is a tuple of (query, reference)
        corpus: the corpus the recordings come from
        logger: logger instance
    """
    logger.info(f"Generating scenarios in {outdir}/")
    
    # Create output directory
    if os.path.exists(outdir):
        logger.warning(f"Directory {outdir}/ already exists. Deleting and regenerating...")
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    
    cwd = os.getcwd()
    valid_scenarios_count = 0
    
    for i, (query, ref) in enumerate(tqdm(pairs_list, desc="Creating scenarios")):
        # Define source paths
        query_path = os.path.join(cwd, corpus.audio_path(query))
        ref_path = os.path.join(cwd, corpus.audio_path(ref))
        query_annot_path = os.path.join(cwd, corpus.annot_path(query))
        ref_annot_path = os.path.join(cwd, corpus.annot_path(ref))
        
        # Validate all source files
        if not all([
            validate_file(query_path, logger),
            validate_file(ref_path, logger),
            validate_file(query_annot_path, logger),
            validate_file(ref_annot_path, logger)
        ]):
            logger.warning(f"Skipping scenario {query} vs {ref} due to invalid files")
            continue

        scenario_id = f"s{valid_scenarios_count+1}"
        # create scenario directory
        scenario_dir = os.path.join(outdir, scenario_id)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # generate symbolic links for query and reference audio files
        query_link = os.path.join(scenario_dir, "query.wav")
        ref_link = os.path.join(scenario_dir, "ref.wav")
        os.symlink(query_path, query_link)
        os.symlink(ref_path, ref_link)
        
        # generate symbolic links for annotation files
        query_annot_link = os.path.join(scenario_dir, "query.beats")
        ref_annot_link = os.path.join(scenario_dir, "ref.beats")
        os.symlink(query_annot_path, query_annot_link)
        os.symlink(ref_annot_path, ref_annot_link)
        
        # generate text file storing the query and reference
        text = f"{query} {ref}\n"
        with open(os.path.join(scenario_dir, "pair.txt"), "w") as f:
            f.write(text)
            
        valid_scenarios_count += 1
    
    logger.info(f"Generated {valid_scenarios_count} scenarios (skipped {len(pairs_list) - valid_scenarios_count})")

    if valid_scenarios_count == 0:
        # Almost always a missing or not-yet-built corpus. Failing here beats
        # reporting success and leaving an empty scenario directory for the
        # experiment step to find.
        logger.error(
            f"No scenarios generated in {outdir}/. Check that the corpus audio and "
            f"annotations are present; see the README for how to obtain them."
        )
        sys.exit(1)


def save_cfg_files(piece_ids: List[str], pairs_list: List[Tuple[str, str]], config: Dict[str, str], logger: logging.Logger):
    """Persist enumerated piece IDs and pairs under cfg/."""
    cfg_dir = Path(config['train_file']).parent
    cfg_dir.mkdir(parents=True, exist_ok=True)

    with open(config['train_file'], 'wb') as f:
        pickle.dump(piece_ids, f)
    with open(config['pair_file'], 'wb') as f:
        pickle.dump(pairs_list, f)

    logger.info(f"Saved piece list: {config['train_file']}")
    logger.info(f"Saved pairs list: {config['pair_file']}")


def resolve_dataset(config: Dict[str, Any], logger: logging.Logger) -> Tuple[corpora.Corpus, List[str], List[Tuple[str, str]]]:
    """
    Get the corpus, recording list, and pair list for a benchmark.

    Benchmarks that declare 'piece_roots' are enumerated from disk and the
    result is written back to cfg/ for reproducibility. The rest read their
    checked-in lists from cfg/.
    """
    corpus = get_corpus(config)

    if 'piece_roots' in config:
        piece_ids, pairs_list = corpora.build_dataset(corpus, config['piece_roots'], logger)
        save_cfg_files(piece_ids, pairs_list, config, logger)
        return corpus, piece_ids, pairs_list

    for key, label in (('train_file', 'piece list'), ('pair_file', 'pair list')):
        if not os.path.exists(config[key]):
            logger.error(f"{label.capitalize()} not found: {config[key]}")
            logger.error("Please run data preparation first")
            sys.exit(1)

    with open(config['train_file'], 'rb') as f:
        piece_ids = pickle.load(f)
    with open(config['pair_file'], 'rb') as f:
        pairs_list = pickle.load(f)
    return corpus, piece_ids, pairs_list


def compute_chroma_stft_features(piece_ids: List[str], corpus: corpora.Corpus, logger: logging.Logger):
    """Compute and save chroma_stft features for given pieces."""
    chroma_stft_dir = f"{FEAT_DIR}/chroma_stft_norm2"
    os.makedirs(chroma_stft_dir, exist_ok=True)
    
    logger.info(f"Computing chroma_stft features for {len(piece_ids)} pieces")
    
    for piece_id in tqdm(piece_ids, desc="Computing chroma_stft"):
        feat_path = f"{chroma_stft_dir}/{piece_id}.npy"
        os.makedirs(os.path.dirname(feat_path), exist_ok=True)
        
        # Skip if already exists
        if os.path.exists(feat_path):
            logger.debug(f"Skipping {piece_id} - already computed")
            continue
        
        audio_path = corpus.audio_path(piece_id)
        y, sr = lb.load(audio_path)
        chroma_stft_feat = lb.feature.chroma_stft(
            y=y, sr=sr, 
            hop_length=constants.DEFAULT_HOP_LENGTH, 
            center=False, 
            norm=2
        )
        np.save(feat_path, chroma_stft_feat)
    
    logger.info("Chroma STFT features computed")


def compute_match_features(piece_ids: List[str], corpus: corpora.Corpus, logger: logging.Logger):
    """Compute and save match features for given pieces."""
    match_dir = f"{FEAT_DIR}/match"
    os.makedirs(match_dir, exist_ok=True)
    
    logger.info(f"Computing match features for {len(piece_ids)} pieces")
    
    for piece_id in tqdm(piece_ids, desc="Computing match features"):
        feat_path = f"{match_dir}/{piece_id}.npy"
        os.makedirs(os.path.dirname(feat_path), exist_ok=True)
        
        # Skip if already exists
        if os.path.exists(feat_path):
            logger.debug(f"Skipping {piece_id} - already computed")
            continue
        
        audio_path = corpus.audio_path(piece_id)
        match_feat = extract_match_features(audio_path)
        np.save(feat_path, match_feat)
    
    logger.info("Match features computed")


# ============================================================================
# Configuration Management
# ============================================================================

def load_system_config(config_path: Optional[str], systems: List[str], corpus: corpora.Corpus, logger: logging.Logger) -> Dict[str, Dict[str, Any]]:
    """
    Load system configuration from JSON file or use defaults.
    
    Args:
        config_path: Path to JSON configuration file, or None for defaults
        systems: List of system names to configure
        corpus: the corpus being run against, for systems that read audio directly
        logger: Logger instance
    
    Returns:
        Dictionary mapping system names to their configurations
    """
    if config_path:
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    # Use default configurations
    logger.info("Using default system configurations")
    return get_default_configs(systems, corpus)


def path_to_monotonic(path: np.ndarray) -> np.ndarray:
    """
    Convert a warping path to a monotonic path by making the reference (second) row
    non-decreasing via cumulative maximum. Path shape is (2, n_frames); row 0 = query, row 1 = reference.
    """
    path = np.asarray(path)
    out = path.copy()
    out[1, :] = np.maximum.accumulate(path[1, :])
    return out


def get_default_configs(systems: List[str], corpus: corpora.Corpus) -> Dict[str, Dict[str, Any]]:
    """Get default configurations for specified systems."""
    configs = {}
    
    for system in systems:
        if system == 'DTW':
            configs[system] = {
                "steps": constants.DEFAULT_DTW_STEPS.tolist(),
                "weights": constants.DEFAULT_DTW_WEIGHTS.tolist(),
                "feat_dir": f"{FEAT_DIR}/chroma_stft_norm2",
                "sr": constants.DEFAULT_SR,
                "hop_length": constants.DEFAULT_HOP_LENGTH,
                "distance_metric": "cosine"
            }
        elif system in ['SOA', 'SOA_MONOTONIC']:
            configs[system] = {
                "steps": constants.DEFAULT_DTW_STEPS.tolist(),
                "weights": constants.DEFAULT_DTW_WEIGHTS.tolist(),
                "feat_dir": f"{FEAT_DIR}/chroma_stft_norm2",
                "sr": constants.DEFAULT_SR,
                "hop_length": constants.DEFAULT_HOP_LENGTH,
                "norm": True,
                "distance_metric": "cosine",
                "monotonic": system == 'SOA_MONOTONIC'
            }
        elif system == 'MATCH':
            # MATCH reads audio rather than cached features, so it needs the
            # root that this benchmark's piece IDs resolve against.
            configs[system] = {
                "audio_root": corpus.id_audio_root
            }
        elif system == 'OLTW':
            configs[system] = {
                "hop_length": constants.DEFAULT_HOP_LENGTH
            }
        elif system.startswith('OLTW_GLOBAL'):
            # Default OLTW_GLOBAL (Setting A)
            configs[system] = {
                "hop_length": constants.DEFAULT_HOP_LENGTH,
                "feat_dir": f"{FEAT_DIR}/chroma_stft_norm2",
                "sr": constants.DEFAULT_SR,
                "distance_metric": "cosine",
                "c": None,  # set to global
                "DTW_steps": [[1, 0], [0, 1], [1, 1]],
                "DTW_weights": [1, 1, 1],
                "window_steps": [[1, 1], [1, 0], [0, 1]]
            }
        elif system.startswith('OLTW_OURS'):
            configs[system] = {
                "hop_length": constants.DEFAULT_HOP_LENGTH,
                "feat_dir": f"{FEAT_DIR}/chroma_stft_norm2",
                "sr": constants.DEFAULT_SR,
                "distance_metric": "cosine",
                "c": 500,  # constraint window
                "DTW_steps": [[1, 0], [0, 1], [1, 1]],
                "DTW_weights": [1, 1, 1],
                "window_steps": [[1, 1], [1, 0], [0, 1]]
            }
        elif system.startswith('MM_'):
            if system.startswith('MM_ARZT'):
                method = 'arzt'
            elif system.startswith('MM_HMM'):
                method = 'hmm'
            else:
                method = 'dixon'
            configs[system] = {
                "method": method,
                "feat_dir": f"{FEAT_DIR}/chroma_stft_norm2",
                "sr": constants.DEFAULT_SR,
                "hop_length": constants.DEFAULT_HOP_LENGTH,
                "distance_metric": "cosine",
                "window_size": 10,
                # MM_DIXON_RAW is the same follower as MM_DIXON, reporting Dixon's
                # alignment_path unreduced. See matchmaker_worker.READOUTS.
                "readout": "raw" if system.endswith('_RAW') else "reduced"
            }
            if system.startswith('MM_ARZT'):
                configs[system]["step_size"] = 3

    return configs


# ============================================================================
# Command Implementations
# ============================================================================

def cmd_prepare(args, logger: logging.Logger):
    """Prepare scenarios for the specified benchmark."""
    config = BENCHMARK_CONFIGS[args.benchmark]
    corpus, _, pairs_list = resolve_dataset(config, logger)

    generate_scenarios(config['scenarios_dir'], pairs_list, corpus, logger)

    logger.info("Scenario preparation complete")


def cmd_features(args, logger: logging.Logger):
    """Compute features for the specified benchmark and systems."""
    config = BENCHMARK_CONFIGS[args.benchmark]
    corpus, piece_ids, _ = resolve_dataset(config, logger)

    logger.info(f"Processing {len(piece_ids)} pieces for {args.benchmark} benchmark")

    # Compute required features based on systems
    compute_chroma_stft_features(piece_ids, corpus, logger)

    logger.info("Feature computation complete")


def cmd_experiment(args, logger: logging.Logger):
    """Run experiments for the specified benchmark and systems."""
    config = BENCHMARK_CONFIGS[args.benchmark]
    scenarios_dir = config['scenarios_dir']
    exp_dir = config['experiments_dir']
    
    # Check scenarios exist
    if not os.path.exists(scenarios_dir):
        logger.error(f"Scenarios directory not found: {scenarios_dir}")
        logger.error("Please run 'prepare' command first")
        sys.exit(1)
    
    # Load system configurations
    corpus = get_corpus(config)
    system_configs = load_system_config(args.config, args.systems, corpus, logger)

    # MATCH resolves audio paths itself rather than reading cached features, so
    # it needs to know the corpus root. Filled in here as well as in
    # get_default_configs, so that a --config file that omits audio_root still
    # runs instead of raising KeyError deep in the runner.
    if 'MATCH' in system_configs:
        system_configs['MATCH'].setdefault('audio_root', corpus.id_audio_root)
    
    # Ensure SOA runs before SOA_MONOTONIC when both are requested
    systems_order = []
    if 'SOA' in args.systems:
        systems_order.append('SOA')
    if 'SOA_MONOTONIC' in args.systems:
        systems_order.append('SOA_MONOTONIC')
    for s in args.systems:
        if s not in ('SOA', 'SOA_MONOTONIC'):
            systems_order.append(s)
    
    jobs = getattr(args, 'jobs', 1)
    resume = getattr(args, 'resume', False)

    # Thread pools were already capped at import time by _limit_worker_threads().
    if jobs > 1:
        logger.info(f"Running experiments with {jobs} worker processes")

    # Ensure experiment directory exists; clean only the systems being run
    os.makedirs(exp_dir, exist_ok=True)
    for system in systems_order:
        system_dir = os.path.join(exp_dir, system)
        if os.path.exists(system_dir) and not resume:
            logger.info(f"Cleaning experiment directory for {system}: {system_dir}")
            shutil.rmtree(system_dir)
        elif os.path.exists(system_dir):
            logger.info(f"Resuming {system}: keeping existing results in {system_dir}")
        os.makedirs(system_dir, exist_ok=True)
    
    # Run experiments for each system
    for system in systems_order:
        logger.info(f"Running {system} experiments")
        
        if system not in system_configs:
            logger.error(f"No configuration found for system: {system}")
            continue
        
        if system == 'SOA_MONOTONIC':
            # If SOA paths exist, convert them to monotonic instead of recomputing
            soa_dir = os.path.join(exp_dir, 'SOA')
            soa_mono_dir = os.path.join(exp_dir, 'SOA_MONOTONIC')
            scenario_ids = [d for d in os.listdir(scenarios_dir)
                           if os.path.isdir(os.path.join(scenarios_dir, d))]
            converted = 0
            for scenario_id in scenario_ids:
                soa_hyp = os.path.join(soa_dir, scenario_id, 'hyp.npy')
                if os.path.isfile(soa_hyp):
                    path = np.load(soa_hyp)
                    path_mono = path_to_monotonic(path)
                    out_path = os.path.join(soa_mono_dir, scenario_id)
                    os.makedirs(out_path, exist_ok=True)
                    np.save(os.path.join(out_path, 'hyp.npy'), path_mono)
                    converted += 1
            if converted:
                logger.info(f"Generated SOA_MONOTONIC from existing SOA paths for {converted} scenarios")
        
        # Convert lists back to numpy arrays for DTW steps/weights
        kwargs = system_configs[system].copy()
        if 'steps' in kwargs:
            kwargs['steps'] = np.array(kwargs['steps']).reshape((-1, 2))
        if 'weights' in kwargs:
            kwargs['weights'] = np.array(kwargs['weights'])
        if 'DTW_steps' in kwargs:
            kwargs['DTW_steps'] = np.array(kwargs['DTW_steps'])
        if 'window_steps' in kwargs:
            kwargs['window_steps'] = np.array(kwargs['window_steps'])
        
        runner = ExperimentRunner(system, kwargs, logger=logger)
        runner.run_batch(scenarios_dir, exp_dir, jobs=jobs)
        
        logger.info(f"{system} experiments complete")
    
    logger.info("All experiments complete")


def cmd_evaluate(args, logger: logging.Logger):
    """Evaluate experiments for the specified benchmark."""
    config = BENCHMARK_CONFIGS[args.benchmark]
    scenarios_dir = config['scenarios_dir']
    exp_dir = config['experiments_dir']
    eval_dir = config['eval_dir']
    
    # Check experiments exist
    if not os.path.exists(exp_dir):
        logger.error(f"Experiments directory not found: {exp_dir}")
        logger.error("Please run 'experiment' command first")
        sys.exit(1)
    
    # Find all system subdirectories
    systems = [d for d in os.listdir(exp_dir) 
               if os.path.isdir(os.path.join(exp_dir, d))]
    
    if not systems:
        logger.warning(f"No systems found in {exp_dir}")
        return
    
    logger.info(f"Evaluating {len(systems)} systems: {', '.join(systems)}")
    
    # Evaluate each system
    for system in systems:
        logger.info(f"Evaluating {system}")
        system_exp_dir = os.path.join(exp_dir, system)
        system_eval_dir = os.path.join(eval_dir, system)

        eval_tools.eval_alignment_batch(system_exp_dir, scenarios_dir, system_eval_dir, logger=logger)
        logger.info(f"{system} evaluation complete")
    
    logger.info("All evaluations complete")


def cmd_run(args, logger: logging.Logger):
    """Run the full pipeline: prepare -> features -> experiment -> evaluate."""
    logger.info("=" * 80)
    logger.info(f"Running full pipeline for benchmark: {args.benchmark}")
    logger.info(f"Systems: {', '.join(args.systems)}")
    logger.info("=" * 80)
    
    # Step 1: Prepare scenarios
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Preparing scenarios")
    logger.info("=" * 80)
    cmd_prepare(args, logger)
    
    # Step 2: Compute features
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Computing features")
    logger.info("=" * 80)
    cmd_features(args, logger)
    
    # Step 3: Run experiments
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Running experiments")
    logger.info("=" * 80)
    cmd_experiment(args, logger)
    
    # Step 4: Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Evaluating results")
    logger.info("=" * 80)
    cmd_evaluate(args, logger)
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SimRealtimeMazurkaBenchmark Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default settings
  python benchmark.py run --benchmark train_small --systems DTW SOA
  
  # Run full pipeline with custom config
  python benchmark.py run --benchmark train_small --config configs/my_config.json
  
  # Run individual steps
  python benchmark.py prepare --benchmark train_small
  python benchmark.py features --benchmark train_small --systems DTW SOA
  python benchmark.py experiment --benchmark train_small --systems OLTW_GLOBAL --config configs/oltw_config.json
  python benchmark.py evaluate --benchmark train_small
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Generate scenarios')
    prepare_parser.add_argument('--benchmark', required=True, 
                                choices=list(BENCHMARK_CONFIGS),
                                help='Benchmark to prepare')
    
    # Features command
    features_parser = subparsers.add_parser('features', help='Compute features')
    features_parser.add_argument('--benchmark', required=True,
                                 choices=list(BENCHMARK_CONFIGS),
                                 help='Benchmark to compute features for')
    
    # Experiment command
    experiment_parser = subparsers.add_parser('experiment', help='Run experiments')
    experiment_parser.add_argument('--benchmark', required=True,
                                   choices=list(BENCHMARK_CONFIGS),
                                   help='Benchmark to run experiments on')
    experiment_parser.add_argument('--systems', nargs='+', required=True,
                                   help='Systems to run (DTW, SOA, SOA_MONOTONIC, MATCH, OLTW, OLTW_GLOBAL, OLTW_OURS, or custom)')
    experiment_parser.add_argument('--config', type=str,
                                   help='JSON configuration file for system parameters')
    experiment_parser.add_argument('--jobs', type=int, default=1,
                                   help='Number of scenarios to run in parallel (default: 1, serial)')
    experiment_parser.add_argument('--resume', action='store_true',
                                   help='Keep existing results instead of clearing the system directory')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate experiments')
    evaluate_parser.add_argument('--benchmark', required=True,
                                 choices=list(BENCHMARK_CONFIGS),
                                 help='Benchmark to evaluate')
    
    # Run command (full pipeline)
    run_parser = subparsers.add_parser('run', help='Run full pipeline')
    run_parser.add_argument('--benchmark', required=True,
                            choices=list(BENCHMARK_CONFIGS),
                            help='Benchmark to run')
    run_parser.add_argument('--systems', nargs='+', required=True,
                            help='Systems to run (DTW, SOA, SOA_MONOTONIC, MATCH, OLTW, OLTW_GLOBAL, OLTW_OURS, or custom)')
    run_parser.add_argument('--config', type=str,
                            help='JSON configuration file for system parameters')
    run_parser.add_argument('--jobs', type=int, default=1,
                            help='Number of scenarios to run in parallel (default: 1, serial)')
    run_parser.add_argument('--resume', action='store_true',
                            help='Keep existing results instead of clearing the system directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Execute command
        if args.command == 'prepare':
            cmd_prepare(args, logger)
        elif args.command == 'features':
            cmd_features(args, logger)
        elif args.command == 'experiment':
            cmd_experiment(args, logger)
        elif args.command == 'evaluate':
            cmd_evaluate(args, logger)
        elif args.command == 'run':
            cmd_run(args, logger)
    except Exception as e:
        logger.error(f"Error executing command: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
