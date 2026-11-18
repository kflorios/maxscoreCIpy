import numpy as np

class PluginHessianEstimator:
    """
    Analytical plug-in estimator for the Hessian / score derivative matrix 
    matching Cattaneo et al. (2020) Hhat.cpp implementation using bandwidth bw.
    """
    def __init__(self, bw=2.0):
        self.bw = bw

    def _gaussian_kernel_derivative(self, u):
        return -u / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * u**2)

    def estimate(self, beta_hat, X, Y):
        n, d = X.shape
        if d < 2:
            raise ValueError("Hessian estimation requires at least 2 regressors.")
        
        d_free = d - 1
        H = np.zeros((d_free, d_free))
        
        index = np.dot(X, beta_hat)
        X_free = X[:, 1:]  # Exclude first column corresponding to beta_1 = 1
        
        for i in range(n):
            u = index[i] / self.bw
            k_prime = self._gaussian_kernel_derivative(u)
            outer_x = np.outer(X_free[i], X_free[i])
            
            term = (2.0 * Y[i] - 1.0) * (k_prime / (self.bw**2)) * outer_x
            H += term
            
        H = H / n
        return 0.5 * (H + H.T)