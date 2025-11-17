from kalman import KalmanFilter
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

class NOAKalman:
    def __init__(self, F, B, H, Q, R, x0, P0, F2, hop_sec = 512 / 22050, steps = np.array([1, 1, 1, 2, 2, 1]).reshape((-1,2)), weights = np.array([1,1,2]), cost_metric = compute_cosine_distance, ref_start_time = 0):
        """
        Initialize the NOAKalman class
        Inputs:
            F: feature matrix of shape (n_features, n_frames) for the first audio file (query)
            B: control input matrix
            H: observation matrix
            Q: process noise covariance
            R: measurement noise covariance
            x0: initial state vector
            P0: initial state covariance matrix
            F2: feature matrix of shape (n_features, n_frames) for the second audio file (reference)
            hop_sec: hop size in seconds
            steps: step sizes for the DTW algorithm
            weights: weights for the DTW algorithm
            cost_metric: cost metric for the DTW algorithm
            ref_start_time: time of the first frame of the reference to align to
        """
        # initialize kalman filter
        self.kalman_filter = KalmanFilter(F, B, H, Q, R, x0, P0)
        
        # initalize NOA function call info
        self.path = [[0,0]]
        self.F2 = F2[:, int(ref_start_time / hop_sec):]
        self.ref_length = self.F2.shape[1]
        self.hop_sec = hop_sec
        self.ref_start_time = ref_start_time
        self.steps = steps
        self.weights = weights
        self.cost_metric = cost_metric
        
        # initalize NOA stored info
        self.max_query_length = 2 * self.ref_length
        self.D = np.full((self.max_query_length, self.ref_length), np.inf, dtype=np.float32)
        self.D[0,0] = 0.0  # Set the starting point cost to zero
        self.B = np.full((self.max_query_length, self.ref_length), -1, dtype=np.int32)
        self.i = 1 # current query index
        self.dn, self.dm = self.steps[:, 0], self.steps[:, 1]
        
    def noa_update(self, f1):
        """
        Updates NOA alignment info with new query feature frame f1
        Inputs:
            f1: new query feature frame
        Output:
            best_j: best index in the reference frame from NOA alignment
        """
        if self.path[-1][1] >= self.ref_length - 1:
            return self.ref_length - 1
        
        costs = self.cost_metric(f1, self.F2)
        best_j = update_alignment_row_numba_norm(
            self.i, costs, self.D, self.B, self.dn, self.dm, self.weights, self.ref_length
        )
        return best_j
    
    def kalman_update(self, best_j):
        """
        Updates Kalman filter with new reference feature frame f2
        Inputs:
            best_j: best index in the reference frame from NOA alignment
        Output:
            x: updated state vector from Kalman filter
        """
        # Convert best_j to measurement vector format (1D array)
        z = np.array([best_j])
        self.kalman_filter.update(z)
        
        predicted_j = self.kalman_filter.x[0]
        # clamp predicted_j to be between last predicted_j and self.ref_length - 1
        predicted_j = max(self.path[-1][1], min(predicted_j, self.ref_length - 1))
        self.path.append([self.i, predicted_j])
    
    def align(self, F1):
        """
        Aligns the query feature matrix F1 with the reference feature matrix F2
        Inputs:
            F1: query feature matrix
        Outputs:
            path: warping path of shape (2, n_frames) where the first row is the indices of F1 and the second row is the indices of F2
        """
        for i in range(1, F1.shape[1]):
            self.i = i
            # Predict step: predict the next state before getting the measurement
            u = np.zeros(1)  # No control input since we have no acceleration
            self.kalman_filter.predict(u)
            
            # Get measurement from NOA alignment
            best_j = self.noa_update(F1[:, i])
            
            # update with the measurement
            self.kalman_update(best_j)
        return self.path
    
    def get_path(self):
        """
        Returns the path of the NOA alignment
        Outputs:
            path: warping path of shape (2, n_frames) where the first row is the indices of F1 and the second row is the indices of F2
        """
        path = np.array(self.path, dtype=np.float32).T
        path[0, :] *= self.hop_sec
        path[1, :] *= self.hop_sec
        path[1, :] += self.ref_start_time
        return path
    
def alignNOAKalman(F1, F2, 
                   F = np.array([[1, 1], [0, 1]]), 
                   B = np.array([[0.5], [1]]), 
                   H = np.array([[1, 0]]), 
                   Q = np.array([[1, 0], [0, 1]]), 
                   R = np.array([[1]]),
                   sigma_x = 1, sigma_v = 1,
                   outfile = None, steps = np.array([1, 1, 1, 2, 2, 1]).reshape((-1,2)), weights = np.array([1,1,2]), cost_metric = compute_cosine_distance, ref_start_time = 0):
    """
    Aligns the query feature matrix F1 with the reference feature matrix F2 using NOA and Kalman filter
    Inputs:
        F1: query feature matrix
        F2: reference feature matrix
        F: state transition matrix
        B: control input matrix
        H: observation matrix
        Q: process noise covariance
        R: measurement noise covariance
        sigma_x: uncertainty of the state vector
        sigma_v: uncertainty of the speed
        outfile: path to save the output pickle file
        steps: step sizes for the DTW algorithm
        weights: weights for the DTW algorithm
        cost_metric: cost metric for the DTW algorithm
        ref_start_time: time of the first frame of the reference to align to
    Outputs:
        path: warping path of shape (2, n_frames) where the first row is the indices of F1 and the second row is the indices of F2
    """
    x0 = np.array([0, 1])
    P0 = np.array([[sigma_x**2, 0], [0, sigma_v**2]]) # initial state covariance matrix
    noa_kalman = NOAKalman(F, B, H, Q, R, x0, P0, F2 = F2, hop_sec = 512 / 22050, steps = steps, weights = weights, cost_metric = cost_metric, ref_start_time = ref_start_time)
    noa_kalman.align(F1)
    path = noa_kalman.get_path()
    if outfile:
        pickle.dump(path, open(outfile, 'wb'))
    return path