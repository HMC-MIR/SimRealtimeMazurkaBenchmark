import numpy as np

class KalmanFilter:
    """
    Minimal linear Kalman filter that tracks a state vector, its covariance,
    and applies the standard predict/update cycle using the provided system
    matrices (`F`, `B`, `H`) and noise covariances (`Q`, `R`). The state `x`
    and covariance `P` are initialized from `x0`, `P0` and then refined via
    `predict` (time update) followed by `update` (measurement correction).
    """
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F # state transition matrix
        self.B = B # control input matrix
        self.H = H # observation matrix
        self.Q = Q # process noise covariance
        self.R_base = R # base measurement noise covariance
        self.R = R # measurement noise covariance
        self.x = x0 # initial state vector
        self.P = P0 # initial state covariance matrix
        self.S = None # innovation covariance matrix
        self.innovation = None # innovation
        self.K = None # Kalman gain
        
    def predict(self, u):
        """
        Predict the state of the system
        Inputs:
            u: control input
        Outputs:
            x: predicted state
        """
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x
    
    def update(self, z):
        """
        Update the state of the system
        Inputs:
            z: measurement
        Outputs:
            x: updated state
        """
        self.S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R # measurement covariance matrix
        self.innovation = z - np.dot(self.H, self.x) # innovation, which is the difference between the measurement and the predicted measurement
        self.innovation = self.clip_innovation(np.sqrt(self.S[0, 0]), 2) # clip the innovation to be between -2*sigma and 2*sigma
        self.R = adaptive_R(self.R_base, self.innovation[0], self.S[0, 0]) # adaptive measurement noise covariance
        self.S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R # recalculate the measurement covariance matrix
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S)) # Kalman gain
        self.x = self.x + np.dot(self.K, self.innovation) # updated state
        self.P = np.dot(np.eye(self.P.shape[0]) - np.dot(self.K, self.H), self.P) # state covariance matrix
        return self.x
    
    def clip_innovation(self, sigma, n):
        """
        Clip the innovation to be between -n*sigma and n*sigma
        Inputs:
            sigma: standard deviation of the innovation
            n: number of standard deviations to clip to
        Outputs:
            innovation: clipped innovation
        """
        self.innovation = np.array([np.clip(self.innovation[0], -n*sigma, n*sigma)])
        return self.innovation
        
def adaptive_R(R_base, nu, S, tau=3.0, inflation=50.0):
    """
    Inflate R when innovation is too large.
    """
    threshold = tau * np.sqrt(S)
    if np.abs(nu) > threshold:
        return R_base * inflation
    else:
        return R_base
