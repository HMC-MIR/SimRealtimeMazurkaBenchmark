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
from typing import List, Dict, Any, Optional

import numpy as np
import librosa as lb
from tqdm import tqdm

# Import local modules
import utils.constants as constants
from utils.experiments import ExperimentRunner
from utils.match_features import extract_match_features
import eval_tools


# ============================================================================
# Configuration and Constants
# ============================================================================

BENCHMARK_CONFIGS = {
    'train_small': {
        'train_file': 'cfg/mazurkas.train.pkl',
        'pair_file': 'cfg/mazurkas.train_pairs.pkl',
        'scenarios_dir': 'scenarios',
        'experiments_dir': 'experiments',
        'eval_dir': 'eval',
    },
    'train': {
        'train_file': 'cfg/mazurkas.train_large.pkl',
        'pair_file': 'cfg/mazurkas.train_pairs_large.pkl',
        'scenarios_dir': 'scenarios_train',
        'experiments_dir': 'experiments_train',
        'eval_dir': 'eval_train',
    },
    'test': {
        'train_file': 'cfg/mazurkas.test.pkl',
        'pair_file': 'cfg/mazurkas.test_pairs.pkl',
        'scenarios_dir': 'scenarios_test',
        'experiments_dir': 'experiments_test',
        'eval_dir': 'eval_test',
    },
}

AUDIO_ROOT = "Chopin_Mazurkas/wav_22050_mono/Chopin_Op017No4"
ANNOT_ROOT = "Chopin_Mazurkas/annotations_beat/Chopin_Op017No4"
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


def generate_scenarios(outdir: str, pairs_list: List[tuple], logger: logging.Logger):
    """
    Generate all possible alignment scenarios for a given output directory.
    
    Args:
        outdir: output directory
        pairs_list: list of pairs of recordings to process. Each pair is a tuple of (query, reference)
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
        query_path = os.path.join(cwd, AUDIO_ROOT, f"{query}.wav")
        ref_path = os.path.join(cwd, AUDIO_ROOT, f"{ref}.wav")
        query_annot_path = os.path.join(cwd, ANNOT_ROOT, f"{query}.beat")
        ref_annot_path = os.path.join(cwd, ANNOT_ROOT, f"{ref}.beat")
        
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


def compute_chroma_stft_features(piece_ids: List[str], logger: logging.Logger):
    """Compute and save chroma_stft features for given pieces."""
    chroma_stft_dir = f"{FEAT_DIR}/chroma_stft_norm2"
    os.makedirs(chroma_stft_dir, exist_ok=True)
    
    logger.info(f"Computing chroma_stft features for {len(piece_ids)} pieces")
    
    for piece_id in tqdm(piece_ids, desc="Computing chroma_stft"):
        feat_path = f"{chroma_stft_dir}/{piece_id}.npy"
        
        # Skip if already exists
        if os.path.exists(feat_path):
            logger.debug(f"Skipping {piece_id} - already computed")
            continue
        
        audio_path = f"{AUDIO_ROOT}/{piece_id}.wav"
        y, sr = lb.load(audio_path)
        chroma_stft_feat = lb.feature.chroma_stft(
            y=y, sr=sr, 
            hop_length=constants.DEFAULT_HOP_LENGTH, 
            center=False, 
            norm=2
        )
        np.save(feat_path, chroma_stft_feat)
    
    logger.info("Chroma STFT features computed")


def compute_match_features(piece_ids: List[str], logger: logging.Logger):
    """Compute and save match features for given pieces."""
    match_dir = f"{FEAT_DIR}/match"
    os.makedirs(match_dir, exist_ok=True)
    
    logger.info(f"Computing match features for {len(piece_ids)} pieces")
    
    for piece_id in tqdm(piece_ids, desc="Computing match features"):
        feat_path = f"{match_dir}/{piece_id}.npy"
        
        # Skip if already exists
        if os.path.exists(feat_path):
            logger.debug(f"Skipping {piece_id} - already computed")
            continue
        
        audio_path = f"{AUDIO_ROOT}/{piece_id}.wav"
        match_feat = extract_match_features(audio_path)
        np.save(feat_path, match_feat)
    
    logger.info("Match features computed")


# ============================================================================
# Configuration Management
# ============================================================================

def load_system_config(config_path: Optional[str], systems: List[str], logger: logging.Logger) -> Dict[str, Dict[str, Any]]:
    """
    Load system configuration from JSON file or use defaults.
    
    Args:
        config_path: Path to JSON configuration file, or None for defaults
        systems: List of system names to configure
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
    return get_default_configs(systems)


def get_default_configs(systems: List[str]) -> Dict[str, Dict[str, Any]]:
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
        elif system in ['NOA', 'NOA_MONOTONIC']:
            configs[system] = {
                "steps": constants.DEFAULT_DTW_STEPS.tolist(),
                "weights": constants.DEFAULT_DTW_WEIGHTS.tolist(),
                "feat_dir": f"{FEAT_DIR}/chroma_stft_norm2",
                "sr": constants.DEFAULT_SR,
                "hop_length": constants.DEFAULT_HOP_LENGTH,
                "norm": True,
                "distance_metric": "cosine",
                "monotonic": system == 'NOA_MONOTONIC'
            }
        elif system == 'MATCH':
            configs[system] = {
                "audio_root": AUDIO_ROOT
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
    
    return configs


# ============================================================================
# Command Implementations
# ============================================================================

def cmd_prepare(args, logger: logging.Logger):
    """Prepare scenarios for the specified benchmark."""
    if args.benchmark == 'test':
        logger.error("Test benchmark not yet implemented")
        sys.exit(1)
    
    config = BENCHMARK_CONFIGS[args.benchmark]
    
    # Load pair list
    pair_file = config['pair_file']
    if not os.path.exists(pair_file):
        logger.error(f"Pair file not found: {pair_file}")
        logger.error("Please run data preparation first to create training pairs")
        sys.exit(1)
    
    with open(pair_file, 'rb') as f:
        pairs_list = pickle.load(f)
    
    # Generate scenarios
    generate_scenarios(config['scenarios_dir'], pairs_list, logger)
    
    logger.info("Scenario preparation complete")


def cmd_features(args, logger: logging.Logger):
    """Compute features for the specified benchmark and systems."""
    if args.benchmark == 'test':
        logger.error("Test benchmark not yet implemented")
        sys.exit(1)
    
    config = BENCHMARK_CONFIGS[args.benchmark]
    
    # Load piece IDs
    train_file = config['train_file']
    if not os.path.exists(train_file):
        logger.error(f"Training file not found: {train_file}")
        logger.error("Please run data preparation first to create training set")
        sys.exit(1)
    
    with open(train_file, 'rb') as f:
        piece_ids = pickle.load(f)
    
    logger.info(f"Processing {len(piece_ids)} pieces for {args.benchmark} benchmark")
    
    # Compute required features based on systems
    compute_chroma_stft_features(piece_ids, logger)
    
    logger.info("Feature computation complete")


def cmd_experiment(args, logger: logging.Logger):
    """Run experiments for the specified benchmark and systems."""
    if args.benchmark == 'test':
        logger.error("Test benchmark not yet implemented")
        sys.exit(1)
    
    config = BENCHMARK_CONFIGS[args.benchmark]
    scenarios_dir = config['scenarios_dir']
    exp_dir = config['experiments_dir']
    
    # Check scenarios exist
    if not os.path.exists(scenarios_dir):
        logger.error(f"Scenarios directory not found: {scenarios_dir}")
        logger.error("Please run 'prepare' command first")
        sys.exit(1)
    
    # Load system configurations
    system_configs = load_system_config(args.config, args.systems, logger)
    
    # Clean and create experiment directories
    logger.info(f"Cleaning experiment directory: {exp_dir}")
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    
    for system in args.systems:
        os.makedirs(os.path.join(exp_dir, system), exist_ok=True)
    
    # Run experiments for each system
    for system in args.systems:
        logger.info(f"Running {system} experiments")
        
        if system not in system_configs:
            logger.error(f"No configuration found for system: {system}")
            continue
        
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
        runner.run_batch(scenarios_dir, exp_dir)
        
        logger.info(f"{system} experiments complete")
    
    logger.info("All experiments complete")


def cmd_evaluate(args, logger: logging.Logger):
    """Evaluate experiments for the specified benchmark."""
    if args.benchmark == 'test':
        logger.error("Test benchmark not yet implemented")
        sys.exit(1)
    
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
  python benchmark.py run --benchmark train_small --systems DTW NOA
  
  # Run full pipeline with custom config
  python benchmark.py run --benchmark train_small --config configs/my_config.json
  
  # Run individual steps
  python benchmark.py prepare --benchmark train_small
  python benchmark.py features --benchmark train_small --systems DTW NOA
  python benchmark.py experiment --benchmark train_small --systems OLTW_GLOBAL --config configs/oltw_config.json
  python benchmark.py evaluate --benchmark train_small
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Generate scenarios')
    prepare_parser.add_argument('--benchmark', required=True, 
                                choices=['train_small', 'train', 'test'],
                                help='Benchmark to prepare')
    
    # Features command
    features_parser = subparsers.add_parser('features', help='Compute features')
    features_parser.add_argument('--benchmark', required=True,
                                 choices=['train_small', 'train', 'test'],
                                 help='Benchmark to compute features for')
    
    # Experiment command
    experiment_parser = subparsers.add_parser('experiment', help='Run experiments')
    experiment_parser.add_argument('--benchmark', required=True,
                                   choices=['train_small', 'train', 'test'],
                                   help='Benchmark to run experiments on')
    experiment_parser.add_argument('--systems', nargs='+', required=True,
                                   help='Systems to run (DTW, NOA, NOA_MONOTONIC, MATCH, OLTW, OLTW_GLOBAL, or custom)')
    experiment_parser.add_argument('--config', type=str,
                                   help='JSON configuration file for system parameters')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate experiments')
    evaluate_parser.add_argument('--benchmark', required=True,
                                 choices=['train_small', 'train', 'test'],
                                 help='Benchmark to evaluate')
    
    # Run command (full pipeline)
    run_parser = subparsers.add_parser('run', help='Run full pipeline')
    run_parser.add_argument('--benchmark', required=True,
                            choices=['train_small', 'train', 'test'],
                            help='Benchmark to run')
    run_parser.add_argument('--systems', nargs='+', required=True,
                            help='Systems to run (DTW, NOA, NOA_MONOTONIC, MATCH, OLTW, OLTW_GLOBAL, or custom)')
    run_parser.add_argument('--config', type=str,
                            help='JSON configuration file for system parameters')
    
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
