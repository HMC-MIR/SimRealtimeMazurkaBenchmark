import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from numba import jit, njit
# from baselineDTW import compute_match_distance_frame_to_feature_matrix

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

def alignNOA(F1, F2, outfile = None, steps = np.array([1, 1, 1, 2, 2, 1]).reshape((-1,2)), weights = np.array([1,1,2]), cost_metric = compute_cosine_distance, ref_start_time = 0):
    '''
    Align two feature matrices using NOA
    Inputs:
        F1: feature matrix of shape (n_features, n_frames) for the first audio file (query)
        F2: feature matrix of shape (n_features, n_frames) for the second audio file (reference)
        outfile: path to save the output pickle file
        steps: step sizes for the DTW algorithm
        weights: weights for the DTW algorithm
        ref_start_time: time of the first frame of the reference to align to
    Outputs:
        path: warping path of shape (2, n_frames) where the first row is the indices of F1 and the second row is the indices of F2
    '''
    
    path = [[0,0]] # initialize path from origin
    hop_sec = 512 / 22050
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

    return path

def alignNOA_batch(train_file, chroma_dir, outdir, steps = np.array([1, 1, 1, 2, 2, 1]).reshape((-1,2)), weights = np.array([1,1,2]), cost_metric = compute_cosine_distance):
    '''
    Aligns two directories of chroma features using DTW
    Inputs:
        train_file: path to the pickle file containing the list of file pairs
        chroma_dir: path to the directory containing the chroma features
        outdir: path to the output directory
        steps: step sizes for the DTW algorithm
        weights: weights for the DTW algorithm
    '''
    
    with open(train_file, 'rb') as f:
        file_pairs = pickle.load(f)

    for counter, pair in tqdm(enumerate(file_pairs), desc = 'Aligning NOA', total=len(file_pairs)):
        try:
            queryid = f"{file_pairs[counter][0]}__{file_pairs[counter][1]}"
            
            # check if queryid is too long, if so, make it shorter
            if len(queryid) > 255:
                queryid = f"{file_pairs[counter][0][:50] + file_pairs[counter][0][-50:]}__{file_pairs[counter][0][:50] + file_pairs[counter][1][-50:]}"

            outfile = (outdir / queryid).with_suffix('.pkl')
            
            # check if file already exists
            if outfile.exists():
                print(f"File {outfile} already exists, skipping...")
                continue
            
            file1 = Path(pair[0])
            file2 = Path(pair[1])

            chroma_path_1 = str(chroma_dir) + '/' + str(file1) + '.pkl'
            chroma_path_2 = str(chroma_dir) + '/' + str(file2) + '.pkl'

            F1 = pickle.load(open(chroma_path_1, 'rb'))
            F2 = pickle.load(open(chroma_path_2, 'rb'))
            
            # dtw cannot handle files that are too long because they max out the RAM
            threshold = 50 * 60 * 22050 / 512
            if F1.shape[1] > threshold or F2.shape[1] > threshold:
                print(f"Skipping {file1} and {file2} because they are too long...")
                continue

            path = alignNOA(F1, F2, outfile = outfile, steps = steps, weights = weights, cost_metric = cost_metric)
            
        except Exception as e:
            print(f"Error in {file1}, {file2}: {e}")
            pass
        
def alignNOA_no_norm(F1, F2, outfile = None, steps = np.array([1, 1, 1, 2, 2, 1]).reshape((-1,2)), weights = np.array([1,1,2]), cost_metric = compute_cosine_distance):
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

    return path

def alignNOA_no_norm_batch(train_file, chroma_dir, outdir, steps = np.array([1, 1, 1, 2, 2, 1]).reshape((-1,2)), weights = np.array([1,1,2]), cost_metric = compute_cosine_distance):
    '''
    Aligns two directories of chroma features using DTW
    Inputs:
        train_file: path to the pickle file containing the list of file pairs
        chroma_dir: path to the directory containing the chroma features
        outdir: path to the output directory
        steps: step sizes for the DTW algorithm
        weights: weights for the DTW algorithm
    '''
    
    with open(train_file, 'rb') as f:
        file_pairs = pickle.load(f)

    for counter, pair in tqdm(enumerate(file_pairs), desc = 'Aligning NOA (no norm)', total=len(file_pairs)):
        try:
            queryid = f"{file_pairs[counter][0]}__{file_pairs[counter][1]}"
            
            # check if queryid is too long, if so, make it shorter
            if len(queryid) > 255:
                queryid = f"{file_pairs[counter][0][:50] + file_pairs[counter][0][-50:]}__{file_pairs[counter][0][:50] + file_pairs[counter][1][-50:]}"

            outfile = (outdir / queryid).with_suffix('.pkl')
            
            # check if file already exists
            if outfile.exists():
                print(f"File {outfile} already exists, skipping...")
                continue
            
            file1 = Path(pair[0])
            file2 = Path(pair[1])

            chroma_path_1 = str(chroma_dir) + '/' + str(file1) + '.pkl'
            chroma_path_2 = str(chroma_dir) + '/' + str(file2) + '.pkl'

            F1 = pickle.load(open(chroma_path_1, 'rb'))
            F2 = pickle.load(open(chroma_path_2, 'rb'))
            
            # dtw cannot handle files that are too long because they max out the RAM
            threshold = 50 * 60 * 22050 / 512
            if F1.shape[1] > threshold or F2.shape[1] > threshold:
                print(f"Skipping {file1} and {file2} because they are too long...")
                continue

            path = alignNOA_no_norm(F1, F2, outfile = outfile, steps = steps, weights = weights, cost_metric = cost_metric)
            
        except Exception as e:
            print(f"Error in {file1}, {file2}: {e}")
            pass