import numpy as np
from scipy.optimize import differential_evolution

class MaxScoreEstimator:
    """
    Computes Manski's Maximum Score Estimator with the first coefficient fixed to 1.
    Parameter vector is parameterized as beta = [1, gamma^T].
    Supports multi-start optimization via n_starts.
    """
    def __init__(self, bounds=(-10.0, 10.0), n_starts=3, seed=None):
        self.bounds = bounds
        self.n_starts = n_starts
        self.seed = seed

    def objective_function(self, gamma, X, Y):
        beta = np.concatenate(([1.0], gamma))
        scores = np.dot(X, beta)
        indicator = (scores >= 0).astype(float)
        val = np.mean((2.0 * Y - 1.0) * indicator)
        return -val # Minimize negative score

    def fit(self, X, Y):
        n, d = X.shape
        if d < 2:
            raise ValueError("Maximum Score Estimator requires at least 2 regressors (d >= 2).")
        
        bounds_list = [self.bounds for _ in range(d - 1)]
        rng = np.random.default_rng(self.seed)
        
        best_fun = np.inf
        best_gamma = None
        
        for i in range(self.n_starts):
            run_seed = int(rng.integers(0, 1000000)) if self.seed is not None else None
            result = differential_evolution(
                self.objective_function,
                bounds=bounds_list,
                args=(X, Y),
                seed=run_seed,
                polish=True,
                strategy='rand1bin',
                popsize=1500,
                maxiter=5000
            )
            if result.fun < best_fun:
                best_fun = result.fun
                best_gamma = result.x
                
        gamma_hat = best_gamma
        return np.concatenate(([1.0], gamma_hat))