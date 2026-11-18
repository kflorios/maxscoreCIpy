import numpy as np
from scipy.optimize import differential_evolution
from .utils import kernel_regression_kappa
from .hessian import PluginHessianEstimator

class ReshapedBootstrap:
    """
    Implements the Cattaneo et al. (2020) Reshaped Bootstrap for Maximum Score Estimator 
    using a unified bandwidth for kernel regression and Hessian estimation, 
    with multi-start optimization via n_starts.
    """
    def __init__(self, B=200, bandwidth=2.0, bounds=(-10.0, 10.0), n_starts=3, seed=None):
        self.B = B
        self.bandwidth = bandwidth
        self.bounds = bounds
        self.n_starts = n_starts
        self.seed = seed

    def fit_bootstrap(self, X, Y, beta_hat):
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        
        # 1. Estimate conditional probability kappa(x) via kernel regression using bandwidth
        kappa_hat = kernel_regression_kappa(X, X, Y, self.bandwidth)
        
        # 2. Compute analytical plug-in Hessian matrix at beta_hat using the same bandwidth
        hess_engine = PluginHessianEstimator(bw=self.bandwidth)
        H_hat = hess_engine.estimate(beta_hat, X, Y)
        
        gamma_hat = beta_hat[1:]
        bootstrap_estimates = []

        for b in range(self.B):
            print("reshaped bootstrap iteration...:",b)
            indices = rng.choice(n, size=n, replace=True)
            X_star = X[indices]
            kappa_star = kappa_hat[indices]
            ###Y_star = rng.binomial(1, kappa_star).astype(float)
            Y_star = Y[indices]
            
            def reshaped_objective(gamma):
                beta = np.concatenate(([1.0], gamma))
                
                # 1. Bootstrap objective M_n^*(beta)
                scores_star = np.dot(X_star, beta)
                ind_star = (scores_star >= 0).astype(float)
                M_star = np.mean((2.0 * Y_star - 1.0) * ind_star)
                
                # 2. Sample objective M_n(beta)
                scores_sample = np.dot(X, beta)
                ind_sample = (scores_sample >= 0).astype(float)
                M_sample = np.mean((2.0 * Y - 1.0) * ind_sample)
                
                # 3. Quadratic correction term: 0.5 * (gamma - gamma_hat)^T H_hat (gamma - gamma_hat)
                diff_gamma = gamma - gamma_hat
                quad_term = 0.5 * np.dot(diff_gamma, np.dot(H_hat, diff_gamma))
                
                # Maximize M_star - M_sample + quad_term -> Minimize negative
                val = M_star - M_sample + quad_term
                return -val
            
            bounds_list = [self.bounds for _ in range(d - 1)]
            
            best_fun = np.inf
            best_gamma = None
            
            for s in range(self.n_starts):
                run_seed = int(rng.integers(0, 1000000))
                result = differential_evolution(
                    reshaped_objective,
                    bounds=bounds_list,
                    seed=run_seed,
                    polish=True,
                    strategy='rand1bin',
                    popsize=1500,
                    maxiter=5000
                )
                if result.fun < best_fun:
                    best_fun = result.fun
                    best_gamma = result.x
            
            gamma_star = best_gamma
            beta_star = np.concatenate(([1.0], gamma_star))
            bootstrap_estimates.append(beta_star)
            
        return np.array(bootstrap_estimates)

    def confidence_intervals(self, beta_hat, bootstrap_estimates, alpha=0.05):
        lower_pct = (alpha / 2) * 100
        upper_pct = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_estimates, lower_pct, axis=0)
        ci_upper = np.percentile(bootstrap_estimates, upper_pct, axis=0)
        
        return np.column_stack((ci_lower, ci_upper))