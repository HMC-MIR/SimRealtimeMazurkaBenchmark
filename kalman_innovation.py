import numpy as np
from scipy.linalg import solve, lstsq
from kalman import KalmanFilter

class InnovationCorrelationKalmanFilter(KalmanFilter):
    """
    Adaptive Kalman Filter using Innovation Correlation Method
    Based on Mehra (1972) - Section V: Innovation Correlation Method
    
    This method:
    1. Computes innovation autocorrelations Γ_k for k=0,1,...,n
    2. Solves for M*H^T (where M is error covariance)
    3. Estimates optimal Kalman gain K directly
    4. Updates Q and R accordingly
    """
    
    def __init__(self, F, B, H, Q, R, x0, P0, n_lags=None, window_size=50):
        super().__init__(F, B, H, Q, R, x0, P0)
        
        # Determine number of lags to use (typically n = state dimension)
        self.n = F.shape[0]  # state dimension
        self.r = H.shape[0]  # measurement dimension
        self.n_lags = n_lags if n_lags is not None else self.n
        
        # Storage for innovation history
        self.window_size = window_size
        self.innovation_history = []
        
        # Innovation autocorrelations Γ_k for k=0,1,...,n_lags
        self.Gamma = [np.zeros((self.r, self.r)) for _ in range(self.n_lags + 1)]
        
        # Forgetting factor for averaging autocorrelations
        self.beta = 0.95  # higher = more smoothing of autocorrelations
        
        # Flag to track if we have enough data for adaptation
        self.adaptation_ready = False
        self.update_counter = 0
        self.adapt_interval = 10  # Update K, Q, R every N steps
        
    def update(self, z):
        """
        Update with measurement, storing innovation and adapting periodically
        """
        # Standard Kalman update
        x_pred = self.x.copy()
        
        self.S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        self.innovation = z - np.dot(self.H, self.x)
        
        # Store innovation for correlation analysis
        self._store_innovation(self.innovation)
        
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S))
        self.x = self.x + np.dot(self.K, self.innovation)
        self.P = np.dot(np.eye(self.P.shape[0]) - np.dot(self.K, self.H), self.P)
        
        self.update_counter += 1
        
        # Perform adaptive estimation periodically
        if self.update_counter >= self.window_size and \
           self.update_counter % self.adapt_interval == 0:
            self._adaptive_update()
        
        return self.x
    
    def _store_innovation(self, innovation):
        """Store innovation in rolling window"""
        self.innovation_history.append(innovation.copy())
        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)
            self.adaptation_ready = True
    
    def _update_innovation_autocorrelations(self):
        """
        Compute innovation autocorrelations Γ_k = E[v_i * v_{i-k}^T]
        Using exponential averaging as in equation (64) of Mehra
        """
        if len(self.innovation_history) < self.n_lags + 1:
            return
        
        N = len(self.innovation_history)
        
        for k in range(self.n_lags + 1):
            # Compute sample autocorrelation at lag k
            gamma_k_sample = np.zeros((self.r, self.r))
            count = 0
            
            for i in range(k, N):
                v_i = self.innovation_history[i].reshape(-1, 1)
                v_ik = self.innovation_history[i - k].reshape(-1, 1)
                gamma_k_sample += np.dot(v_i, v_ik.T)
                count += 1
            
            if count > 0:
                gamma_k_sample /= count
                
                # Exponential averaging (forgetting factor)
                self.Gamma[k] = self.beta * self.Gamma[k] + \
                               (1 - self.beta) * gamma_k_sample
    
    def _estimate_M_HT(self):
        """
        Estimate M*H^T using innovation autocorrelations
        Based on equation (65) in Mehra:
        
        [Γ_1 + H*Φ*K0*R_0  ]       [M_1*H^T]
        [Γ_2 + H*Φ*K0*Γ_1 + H*Φ²*K0*R_0] = A * [M_1*H^T]
        [      ...          ]
        [Γ_n + ...          ]
        
        where K0 is the current (suboptimal) gain
        """
        # Build the system of equations
        # Simplified version: Γ_k ≈ H*Φ^k*M_1*H^T for k > 0
        
        # Construct matrix A from equation (46): A^T = [H^T, (Φ*H)^T, ..., (Φ^n*H)^T]
        A_rows = []
        Phi_power = np.eye(self.n)
        
        for k in range(self.n_lags + 1):
            A_rows.append(np.dot(Phi_power, self.H.T))
            Phi_power = np.dot(Phi_power, self.F)
        
        A = np.vstack(A_rows)  # Shape: (r*(n_lags+1), n)
        
        # Stack the autocorrelations (right-hand side)
        # Need to account for K0 terms as in equation (64)
        b_rows = []
        
        for k in range(1, self.n_lags + 1):
            # Γ_k adjusted for current suboptimal gain K0
            adjusted_gamma = self.Gamma[k].copy()
            
            # Add correction terms: H*Φ^(k-j)*K0*Γ_{j-1} for j=1 to k
            Phi_power = np.eye(self.n)
            for j in range(1, k + 1):
                Phi_power_j = np.linalg.matrix_power(self.F, k - j)
                correction = np.dot(self.H, np.dot(Phi_power_j, 
                                   np.dot(self.K, self.Gamma[j - 1])))
                adjusted_gamma += correction
            
            b_rows.append(adjusted_gamma)
        
        if len(b_rows) == 0:
            return None
        
        # Solve for M_1*H^T using least squares
        # Each Γ_k is r x r, so we solve for each column
        M_HT = np.zeros((self.n, self.r))
        
        for col in range(self.r):
            b_col = np.vstack([gamma[:, col].reshape(-1, 1) 
                              for gamma in b_rows])
            
            # Solve A^T * (M_HT[:, col]) = b_col
            try:
                solution, _, _, _ = lstsq(A[self.r:, :].T, b_col)
                M_HT[:, col] = solution.flatten()
            except:
                # If solution fails, keep current estimate
                pass
        
        return M_HT
    
    def _estimate_optimal_gain(self, M_HT):
        """
        Estimate optimal Kalman gain from M*H^T
        Based on equations (54) and (70) in Mehra:
        K = (M_1*H^T + δM*H^T) * (Γ_0 + H*δM*H^T)^{-1}
        
        Simplified: K ≈ M*H^T * Γ_0^{-1}
        """
        if M_HT is None:
            return self.K
        
        try:
            # Use Γ_0 as estimate of innovation covariance
            K_new = np.dot(M_HT, np.linalg.inv(self.Gamma[0]))
            return K_new
        except:
            return self.K
    
    def _estimate_R(self, M_HT):
        """
        Estimate measurement noise covariance R
        Based on equation (66): R = Γ_0 - H*M_1*H^T
        """
        if M_HT is None:
            return self.R
        
        try:
            R_new = self.Gamma[0] - np.dot(self.H, M_HT)
            
            # Ensure R is positive definite
            eigenvalues = np.linalg.eigvals(R_new)
            if np.all(eigenvalues > 0):
                return R_new
            else:
                # If not positive definite, add small regularization
                return R_new + 1e-6 * np.eye(self.r)
        except:
            return self.R
    
    def _estimate_Q(self, K_new, M_new):
        """
        Estimate process noise covariance Q
        Based on equation (67): M_1 = Φ*(M_1 - K*H*M_1)*Φ^T + Q
        Solving for Q: Q = M_1 - Φ*(M_1 - K*H*M_1)*Φ^T
        """
        try:
            Q_new = M_new - np.dot(self.F, 
                                   np.dot(M_new - np.dot(K_new, np.dot(self.H, M_new)),
                                          self.F.T))
            
            # Ensure Q is positive semi-definite
            eigenvalues = np.linalg.eigvals(Q_new)
            if np.all(eigenvalues >= -1e-10):  # Allow small numerical errors
                return np.maximum(Q_new, 0)  # Force non-negative
            else:
                return self.Q
        except:
            return self.Q
    
    def _adaptive_update(self):
        """
        Perform the complete adaptive update:
        1. Update innovation autocorrelations
        2. Estimate M*H^T
        3. Estimate optimal gain K
        4. Estimate R and Q
        """
        if not self.adaptation_ready:
            return
        
        # Step 1: Update autocorrelations
        self._update_innovation_autocorrelations()
        
        # Step 2: Estimate M*H^T
        M_HT = self._estimate_M_HT()
        
        if M_HT is not None:
            # Step 3: Estimate optimal gain
            K_new = self._estimate_optimal_gain(M_HT)
            
            # Step 4: Estimate R
            R_new = self._estimate_R(M_HT)
            
            # Step 5: Estimate M (error covariance)
            # M ≈ M_HT * H * (H^T * H)^{-1}
            try:
                HTH_inv = np.linalg.inv(np.dot(self.H.T, self.H))
                M_new = np.dot(M_HT, np.dot(self.H, HTH_inv))
            except:
                M_new = self.P
            
            # Step 6: Estimate Q
            Q_new = self._estimate_Q(K_new, M_new)
            
            # Update with smoothing to avoid instability
            gamma = 0.7  # Smoothing factor for updates
            self.K = gamma * self.K + (1 - gamma) * K_new
            self.R = gamma * self.R + (1 - gamma) * R_new
            self.Q = gamma * self.Q + (1 - gamma) * Q_new