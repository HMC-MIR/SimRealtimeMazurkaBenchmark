import os
import logging
import subprocess

import numpy as np
from numba import jit, prange
from hmc_mir.align import dtw
from tqdm import tqdm
import vamp
import pandas as pd

from noa import compute_cosine_distance, compute_euclidean_distance
from utils.oltw import online_processing
from online_alignment import run_offline_oltw, run_offline_noa

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

class ExperimentRunner:
    def __init__(self, exp_type, kwargs, logger=None):
        """
        exp_type: experiment to run. Currently accepts DTW, NOA, or MATCH
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
        
        # check if out_path exists. if so, skip
        if os.path.exists(out_path):
            print(f"Skipping {out_path} because it already exists")
            return
        
        # generate out_path
        os.makedirs(out_path, exist_ok=True)
        
        # run experiment
        if self.exp_type == "DTW":
            self.run_dtw(scenarios_dir, out_path)
        elif self.exp_type == "NOA" or self.exp_type == "NOA_MONOTONIC":
            self.run_noa(scenarios_dir, out_path)
        elif self.exp_type == "MATCH":
            self.run_match(scenarios_dir, out_path)
        elif self.exp_type == "OLTW":
            self.run_oltw(scenarios_dir, out_path)
        elif "OLTW_" in self.exp_type:
            self.run_oltw_global(scenarios_dir, out_path)
        else:
            raise ValueError(f"Invalid experiment type: {self.exp_type}")
            
    def run_batch(self, scenarios_root, out_dir):
        """
        Runs experiments for all scenarios under scenarios_root.
        """
        if not os.path.isdir(scenarios_root):
            raise ValueError(f"{scenarios_root} is not a directory")
        
        for scenario_dir in tqdm(os.listdir(scenarios_root)):
            scenario_path = os.path.join(scenarios_root, scenario_dir)
            if os.path.isdir(scenario_path):
                try:
                    self.run(scenario_path, out_dir)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error running experiment for {scenario_path}: {e}", exc_info=True)
                    else:
                        print(f"Error running experiment for {scenario_path}: {e}")
                    continue
                
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
        
        
    def run_noa(self, scenarios_dir, out_path, monotonic = False):
        """
        Runs NOA experiment for the given scenario and stores results to output path.
        """
        # generate out_path
        os.makedirs(out_path, exist_ok=True)
        
        # load query and reference features
        query_feat, reference_feat = self.load_feat(scenarios_dir)
        
        # run NOA
        norm = self.kwargs['norm']
        monotonic = self.kwargs['monotonic']
        wp = run_offline_noa(reference_feat, query_feat, cost_metric = self.kwargs['distance_metric'], monotonic = monotonic, normalize = norm)
        
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

        # run NOA
        wp = run_offline_oltw(reference_feat, query_feat, c=self.kwargs['c'], DTW_steps=DTW_steps, window_steps=window_steps, DTW_weights=DTW_weights, cost_metric=self.kwargs['distance_metric'])
        
        # convert to seconds
        hop_sec = self.kwargs['hop_length'] / self.kwargs['sr']
        wp_sec = wp * hop_sec
        
        # store result
        np.save(os.path.join(out_path, "hyp.npy"), wp_sec)
        
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