import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from numba import jit, njit

@njit(cache=True)
def compute_cosine_distance(feature_row, reference_features):
    """Compute cosine distance between normalized feature vectors.

    Assumes both feature_row and reference_features are already normalized (unit vectors).
    For normalized vectors, cosine distance = 1 - dot_product.
    """
    costs = np.empty(reference_features.shape[1], dtype=np.float32)

    for j in range(reference_features.shape[1]):
        ref_col = reference_features[:, j]
        dot_product = np.sum(feature_row * ref_col)
        costs[j] = 1.0 - dot_product

    return costs

@njit(cache=True)
def compute_euclidean_distance(feature_row, reference_features):
    """Compute Euclidean distance between feature vectors.
    """
    costs = np.empty(reference_features.shape[1], dtype=np.float32)
    for j in range(reference_features.shape[1]):
        ref_col = reference_features[:, j]
        costs[j] = np.sqrt(np.sum((feature_row - ref_col) ** 2))
    return costs

@njit(cache=True)
def update_alignment_row_numba_norm(i, costs, D, B, dn, dm, dw, ref_length):
    """
    Update alignment row using numba for optimized performance.

    Args:
        i (int): Current row index in the alignment matrix.
        costs (np.ndarray): Cost vector for the current feature row.
        D (np.ndarray): Alignment cost matrix.
        B (np.ndarray): Backtrace matrix.
        dn (np.ndarray): Row step sizes.
        dm (np.ndarray): Column step sizes.
        dw (np.ndarray): Weights for each step.
        ref_length (int): Length of the reference features.
    """
    best_j = 0
    best_cost = np.inf

    for j in range(min(costs.shape[0], ref_length)):
        best_step_cost = np.inf
        best_step_cost_norm = np.inf
        best_step = -1

        for k, (di, dj, w) in enumerate(zip(dn, dm, dw)):
            prev_i, prev_j = i - di, j - dj

            if prev_i < 0 or prev_j < 0 or prev_j >= ref_length:
                continue

            cur_cost = D[prev_i, prev_j] + costs[j] * w
            norm_cost = cur_cost / (i + 1 + j + 1)  # Normalize

            if norm_cost < best_step_cost_norm:
                best_step_cost_norm = norm_cost
                best_step = k
                best_step_cost = cur_cost

        if best_step != -1:
            D[i, j] = best_step_cost
            B[i, j] = best_step

            if best_step_cost_norm < best_cost:
                best_cost = best_step_cost_norm
                best_j = j

    return best_j

def alignNOA(F1, F2, outfile = None, steps = np.array([1, 1, 1, 2, 2, 1]).reshape((-1,2)), weights = np.array([1,1,2]), cost_metric = compute_cosine_distance, ref_start_time = 0, return_D = False, hop_sec = 512 / 22050, monotonous = False):
    """
    Align two feature matrices using NOA
    Inputs:
        F1: feature matrix of shape (n_features, n_frames) for the first audio file (query)
        F2: feature matrix of shape (n_features, n_frames) for the second audio file (reference)
        outfile: path to save the output pickle file
        steps: step sizes for the DTW algorithm
        weights: weights for the DTW algorithm
        ref_start_time: time of the first frame of the reference to align to
        return_D: whether to return the cost matrix D
        hop_sec: hop size in seconds
    Outputs:
        path: warping path of shape (2, n_frames) where the first row is the indices of F1 and the second row is the indices of F2
    """
    
    path = [[0,0]] # initialize path from origin
    F2 = F2[:, int(ref_start_time / hop_sec):]
    ref_length = F2.shape[1]
    dn, dm = steps[:, 0], steps[:, 1]
    
    # initalize matrices
    max_query_length = 2 * ref_length
    D = np.full((max_query_length, ref_length), np.inf, dtype=np.float32)
    D[0,0] = 0.0  # Set the starting point cost to zero
    B = np.full((max_query_length, ref_length), -1, dtype=np.int32)
    
    for i in range(1, F1.shape[1]):
        if path[-1][1] >= ref_length - 1:
            break
        
        costs = cost_metric(F1[:, i], F2)
        best_j = update_alignment_row_numba_norm(
            i, costs, D, B, dn, dm, weights, ref_length
        )
        
        if monotonous:
            best_j = max(best_j, path[-1][1])
        
        path.append([i, best_j])
        
    # convert path to numpy array
    path = np.array(path, dtype=np.float32).T

    # convert to seconds
    path[0, :] *= hop_sec
    path[1, :] *= hop_sec

    # add ref_start_time to path
    path[1, :] += ref_start_time
    
    if outfile:
        pickle.dump(path, open(outfile, 'wb'))
        
    if return_D:
        return path, D
    else:
        return path

@njit(cache=True)
def update_alignment_row_numba(i, costs, D, B, dn, dm, dw, ref_length):
    """
    Update alignment row using numba for optimized performance.

    Args:
        i (int): Current row index in the alignment matrix.
        costs (np.ndarray): Cost vector for the current feature row.
        D (np.ndarray): Alignment cost matrix.
        B (np.ndarray): Backtrace matrix.
        dn (np.ndarray): Row step sizes.
        dm (np.ndarray): Column step sizes.
        dw (np.ndarray): Weights for each step.
        ref_length (int): Length of the reference features.
    """
    best_j = 0
    best_cost = np.inf

    for j in range(min(costs.shape[0], ref_length)):
        best_step_cost = np.inf
        best_step = -1

        for k, (di, dj, w) in enumerate(zip(dn, dm, dw)):
            prev_i, prev_j = i - di, j - dj

            if prev_i < 0 or prev_j < 0 or prev_j >= ref_length:
                continue

            cur_cost = D[prev_i, prev_j] + costs[j] * w

            if cur_cost < best_step_cost:
                best_step_cost = cur_cost
                best_step = k

        if best_step != -1:
            D[i, j] = best_step_cost
            B[i, j] = best_step

            if best_step_cost < best_cost:
                best_cost = best_step_cost
                best_j = j

    return best_j
        
def alignNOA_no_norm(F1, F2, outfile = None, steps = np.array([1, 1, 1, 2, 2, 1]).reshape((-1,2)), weights = np.array([1,1,2]), cost_metric = compute_cosine_distance, return_D = False):
    '''
    Align two feature matrices using NOA without normalization
    Inputs:
        F1: feature matrix of shape (n_features, n_frames) for the first audio file
        F2: feature matrix of shape (n_features, n_frames) for the second audio file
        outfile: path to save the output pickle file
        steps: step sizes for the DTW algorithm
        weights: weights for the DTW algorithm
    Outputs:
        path: warping path of shape (2, n_frames) where the first row is the indices of F1 and the second row is the indices of F2
    '''
    
    path = [[0,0]] # initialize path from origin
    ref_length = F2.shape[1]
    dn, dm = steps[:, 0], steps[:, 1]
    
    # initalize matrices
    max_query_length = 2 * ref_length
    D = np.full((max_query_length, ref_length), np.inf, dtype=np.float32)
    D[0,0] = 0.0  # Set the starting point cost to zero
    B = np.full((max_query_length, ref_length), -1, dtype=np.int32)
    
    for i in range(1, F1.shape[1]):
        if path[-1][1] >= ref_length - 1:
            break
        
        costs = cost_metric(F1[:, i], F2)
        best_j = update_alignment_row_numba(
            i, costs, D, B, dn, dm, weights, ref_length
        )
        
        path.append([i, best_j])
        
    # convert path to numpy array
    path = np.array(path, dtype=np.int32).T
    
    if outfile:
        pickle.dump(path, open(outfile, 'wb'))

    if return_D:
        return path, D
    else:
        return path