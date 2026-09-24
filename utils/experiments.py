import os
import logging
import multiprocessing
import subprocess

import numpy as np
from numba import jit, prange
from hmc_mir.align import dtw
from tqdm import tqdm
import pandas as pd

from utils.oltw import online_processing
from utils.matchmaker_baseline import run_matchmaker_alignment
from online_alignment import run_offline_oltw, run_offline_soa

@jit(nopython=True, parallel=True)
def cosine_dist(F1, F2):
    '''
    Calculates the pairwise cosine distance matrix between two features matrices.

    Inputs
    F1: the first feature matrix, shape D x N
    F2: the second feature matrix, shape D x M

    Returns a pairwise cost matrix C of shape N x M, where elements indicate cosine distance.
    '''
    F1 = F1.T
    F2 = F2.T
    C = np.zeros((F1.shape[0], F2.shape[0]))
    for row in prange(F1.shape[0]):
        for col in prange(F2.shape[0]):
            C[row, col] = 1 - np.dot(F1[row], F2[col]) / (np.linalg.norm(F1[row]) * np.linalg.norm(F2[col]) + 1e-9)
    return C

@jit(nopython=True, parallel=True)
def euclidean_dist(F1, F2):
    '''
    Calculates the pairwise Euclidean distance matrix between two features matrices.

    Inputs
    F1: the first feature matrix, shape D x N
    F2: the second feature matrix, shape D x M

    Returns a pairwise cost matrix C of shape N x M, where elements indicate Euclidean distance.
    '''
    F1 = F1.T  # Now shape (N, D)
    F2 = F2.T  # Now shape (M, D)
    C = np.zeros((F1.shape[0], F2.shape[0]))
    for row in prange(F1.shape[0]):
        for col in prange(F2.shape[0]):
            diff = F1[row] - F2[col]
            C[row, col] = np.sqrt(np.sum(diff * diff))
    return C

def parse_match_outfile(infile):
    '''
    Parses the MATCH csv output file specifying the estimated alignment.

    Inputs
    infile: filepath to the MATCH csv output file

    Returns a 2xN array indicating the estimated alignment in seconds.
    '''
    d = pd.read_csv(infile, header=None)
    return np.vstack((d.loc[:,1], d.loc[:,2]))

def _run_scenario(task):
    """
    Runs one scenario in a worker process.

    Module level, and rebuilds the runner from (exp_type, kwargs), so that nothing
    unpicklable (such as a logger) has to cross the process boundary.

    Inputs
    task: a tuple of (exp_type, kwargs, scenario_path, out_dir)

    Returns a tuple of (scenario_path, error string or None).
    """
    exp_type, kwargs, scenario_path, out_dir = task
    try:
        ExperimentRunner(exp_type, kwargs).run(scenario_path, out_dir)
        return scenario_path, None
    except Exception as e:
        return scenario_path, repr(e)


class ExperimentRunner:
    def __init__(self, exp_type, kwargs, logger=None):
        """
        exp_type: system key to run, e.g. DTW, SOA, SOA_MONOTONIC, OLTW, OLTW_OURS,
                  OLTW_GLOBAL, MM_DIXON, MM_ARZT (see the README's Systems table)
        kwargs: arguments needed to pass in for the experiment
        logger: optional logger instance
        """
        self.exp_type = exp_type
        self.kwargs = kwargs
        self.logger = logger
        
    def run(self, scenarios_dir, out_dir):
        """
        Runs experiments for the given scenario and stores results to output directory.
        
        Example: scenarios_dir = "scenarios/s1", out_dir = "experiments"
        """
        scenario_id = scenarios_dir.split("/")[-1] # e.g. s1
        out_path = f"{out_dir}/{self.exp_type}/{scenario_id}" # e.g. experiments/DTW/s1
        
        # check if the result already exists. if so, skip
        if os.path.exists(f"{out_path}/hyp.npy"):
            if self.logger:
                self.logger.debug(f"Skipping {out_path} because it already exists")
            return
        
        # generate out_path
        os.makedirs(out_path, exist_ok=True)
        
        # run experiment
        if self.exp_type == "DTW" or self.exp_type.startswith("DTW"):
            self.run_dtw(scenarios_dir, out_path)
        elif self.exp_type in ("SOA", "SOA_MONOTONIC") or self.exp_type.startswith("SOA"):
            self.run_soa(scenarios_dir, out_path)
        elif self.exp_type == "MATCH":
            self.run_match(scenarios_dir, out_path)
        elif self.exp_type == "OLTW":
            self.run_oltw(scenarios_dir, out_path)
        elif self.exp_type.startswith("MM_"):
            self.run_matchmaker(scenarios_dir, out_path)
        elif "OLTW_" in self.exp_type:
            self.run_oltw_global(scenarios_dir, out_path)
        else:
            raise ValueError(f"Invalid experiment type: {self.exp_type}")
            
    def _log_scenario_error(self, scenario_path, error, exc_info=False):
        """
        Reports a per-scenario failure without aborting the batch.
        """
        message = f"Error running experiment for {scenario_path}: {error}"
        if self.logger:
            self.logger.error(message, exc_info=exc_info)
        else:
            print(message)

    def run_batch(self, scenarios_root, out_dir, jobs=1):
        """
        Runs experiments for all scenarios under scenarios_root.

        Inputs
        scenarios_root: directory holding one subdirectory per scenario
        out_dir: directory to write results to
        jobs: number of worker processes. 1 runs serially in this process.
        """
        if not os.path.isdir(scenarios_root):
            raise ValueError(f"{scenarios_root} is not a directory")

        scenario_paths = sorted(
            os.path.join(scenarios_root, d)
            for d in os.listdir(scenarios_root)
            if os.path.isdir(os.path.join(scenarios_root, d))
        )

        if jobs <= 1:
            for scenario_path in tqdm(scenario_paths):
                try:
                    self.run(scenario_path, out_dir)
                except Exception as e:
                    self._log_scenario_error(scenario_path, e, exc_info=True)
                    continue
            return

        # Scenarios are independent: each writes only its own hyp.npy and reads
        # shared read-only feature files, so they parallelise without coordination.
        # chunksize=1 because scenario durations vary widely.
        tasks = [(self.exp_type, self.kwargs, path, out_dir) for path in scenario_paths]
        with multiprocessing.Pool(processes=jobs) as pool:
            for scenario_path, error in tqdm(
                pool.imap_unordered(_run_scenario, tasks, chunksize=1), total=len(tasks)
            ):
                if error is not None:
                    self._log_scenario_error(scenario_path, error)
                
    def load_feat(self, scenarios_dir):
        """
        Loads features for the given scenario.
        """
        # load query and reference
        with open(os.path.join(scenarios_dir, "pair.txt"), "r") as f:
            query, reference = f.read().split()
            
        # load features
        query_feat_path = f"{self.kwargs['feat_dir']}/{query}.npy"
        reference_feat_path = f"{self.kwargs['feat_dir']}/{reference}.npy"
        query_feat = np.load(query_feat_path)
        reference_feat = np.load(reference_feat_path)
        
        return query_feat, reference_feat
                
    def run_dtw(self, scenarios_dir, out_path):
        """
        Runs DTW experiment for the given scenario and stores results to output path.
        """
        # generate out_path
        os.makedirs(out_path, exist_ok=True)
        
        # load query and reference features
        query_feat, reference_feat = self.load_feat(scenarios_dir)
        
        # run DTW
        if self.kwargs['distance_metric'] == 'cosine':
            C = cosine_dist(query_feat, reference_feat)
        elif self.kwargs['distance_metric'] == 'euclidean':
            C = euclidean_dist(query_feat, reference_feat)
        else:
            raise ValueError(f"Invalid distance metric: {self.kwargs['distance_metric']}")
        _, _, wp = dtw.dtw(C, self.kwargs['steps'], self.kwargs['weights'], True)
        
        # store result
        hop_sec = self.kwargs['hop_length'] / self.kwargs['sr']
        wp_sec = wp * hop_sec
        np.save(os.path.join(out_path, "hyp.npy"), wp_sec)
        
        
    def run_soa(self, scenarios_dir, out_path):
        """
        Runs SOA experiment for the given scenario and stores results to output path.
        """
        # generate out_path
        os.makedirs(out_path, exist_ok=True)
        
        # load query and reference features
        query_feat, reference_feat = self.load_feat(scenarios_dir)
        
        # run SOA
        norm = self.kwargs['norm']
        monotonic = self.kwargs['monotonic']
        wp = run_offline_soa(reference_feat, query_feat, steps = self.kwargs['steps'], weights = self.kwargs['weights'], cost_metric = self.kwargs['distance_metric'], monotonic = monotonic, normalize = norm)
        
        # convert to seconds
        hop_sec = self.kwargs['hop_length'] / self.kwargs['sr']
        wp_sec = wp * hop_sec
        
        # store result
        np.save(os.path.join(out_path, "hyp.npy"), wp_sec)
        
    def run_oltw_global(self, scenarios_dir, out_path):
        """
        Runs global OLTW experiment for the given scenario and stores results to output path.
        """
        # generate out_path
        os.makedirs(out_path, exist_ok=True)
        
        # load query and reference features
        query_feat, reference_feat = self.load_feat(scenarios_dir)
        
        # parse steps and weights for window and transition
        DTW_steps = self.kwargs['DTW_steps']
        window_steps = self.kwargs['window_steps']
        DTW_weights = self.kwargs['DTW_weights']

        # run OLTW
        wp = run_offline_oltw(reference_feat, query_feat, c=self.kwargs['c'], steps=DTW_steps, window_steps=window_steps, weights=DTW_weights, cost_metric=self.kwargs['distance_metric'])
        
        # convert to seconds
        hop_sec = self.kwargs['hop_length'] / self.kwargs['sr']
        wp_sec = wp * hop_sec
        
        # store result
        np.save(os.path.join(out_path, "hyp.npy"), wp_sec)
        
    def run_matchmaker(self, scenarios_dir, out_path):
        """
        Runs a MatchMaker OLTW baseline for the given scenario and stores results to output path.
        """
        # generate out_path
        os.makedirs(out_path, exist_ok=True)

        # load query and reference
        with open(os.path.join(scenarios_dir, "pair.txt"), "r") as f:
            query, reference = f.read().split()

        # run MatchMaker in its own environment, which writes the alignment directly
        run_matchmaker_alignment(
            ref_feat_path=f"{self.kwargs['feat_dir']}/{reference}.npy",
            query_feat_path=f"{self.kwargs['feat_dir']}/{query}.npy",
            out_file=os.path.join(out_path, "hyp.npy"),
            method=self.kwargs['method'],
            sr=self.kwargs['sr'],
            hop_length=self.kwargs['hop_length'],
            window_size=self.kwargs['window_size'],
            distance_metric=self.kwargs['distance_metric'],
            step_size=self.kwargs.get('step_size'),
            readout=self.kwargs.get('readout', 'reduced'),
        )

    def run_match(self, scenarios_dir, out_path):
        """
        Runs MATCH experiment for the given scenario and stores results to output path.
        """
        # load query and reference
        with open(os.path.join(scenarios_dir, "pair.txt"), "r") as f:
            query, reference = f.read().split()
            
        # load audio files
        query_audio_path = f"{self.kwargs['audio_root']}/{query}.wav"
        reference_audio_path = f"{self.kwargs['audio_root']}/{reference}.wav"
        
        match_align_filepath = f'{out_path}/match_p_pref.out'
        with open(match_align_filepath, 'w') as f:
            subprocess.run(['sonic-annotator', '-d', 'vamp:match-vamp-plugin:match:b_a', '-m', query_audio_path, reference_audio_path, '-w', 'csv', '--csv-stdout'],
                           check=True, stdout=f, stderr=subprocess.DEVNULL)
            
        # store result
        wp= parse_match_outfile(match_align_filepath)
        np.save(os.path.join(out_path, "hyp.npy"), wp)
        
    def run_oltw(self, scenarios_dir, out_path):
        online_processing(scenarios_dir, out_path, self.kwargs['hop_length'])