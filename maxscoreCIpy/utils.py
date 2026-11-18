import numpy as np

def kernel_pdf(u):
    """Standard Gaussian kernel function."""
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)

def kernel_regression_kappa(x_eval, X, Y, h):
    """
    Nonparametric kernel regression estimator for kappa(x) = E[Y | X=x].
    """
    n, d = X.shape
    kappa_vals = np.zeros(x_eval.shape[0])
    
    for idx, x in enumerate(x_eval):
        diff = (X - x) / h
        weights = np.prod(kernel_pdf(diff), axis=1)
        sum_w = np.sum(weights)
        if sum_w < 1e-12:
            kappa_vals[idx] = 0.5
        else:
            kappa_vals[idx] = np.sum(weights * Y) / sum_w
            
    return np.clip(kappa_vals, 0.001, 0.999)