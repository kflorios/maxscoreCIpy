# maxscoreCIpy

Python implementation of Manski's Maximum Score Estimator with **fixed $\beta_1 = 1$**, Cattaneo, Jansson, and Nagasawa (2020) **Reshaped Bootstrap**, analytical Gaussian kernel plug-in Hessian estimation using a unified bandwidth parameter, and multi-start Differential Evolution (`n_starts`).

## Installation

This package requires Python 3.8 or higher (`>=3.8`).

```bash
pip install -e .
```

---

## Quick Start Example

```python
import numpy as np
from maxscoreCIpy import MaxScoreEstimator, ReshapedBootstrap, PluginHessianEstimator

# Generate synthetic data targeting beta = [1.0, 0.5]
np.random.seed(42)
n = 1000
X = np.random.uniform(-1, 1, size=(n, 2))
true_beta = np.array([1.0, 0.5])

latent = np.dot(X, true_beta) + np.random.normal(0, 0.3, size=n)
Y = (latent >= 0).astype(float)

# 1. Point Estimation (first coefficient fixed to 1)
estimator = MaxScoreEstimator(bounds=(-10.0, 10.0), n_starts=3, seed=42)
beta_hat = estimator.fit(X, Y)
print("Estimated Beta (beta_1 = 1 fixed):", beta_hat)

# 2. Analytical Plug-in Hessian Estimation
hessian_calc = PluginHessianEstimator(bw=0.3)
H_plugin = hessian_calc.estimate(beta_hat, X, Y)
print("Analytical Plug-in Hessian Matrix:\n", H_plugin)

# 3. Reshaped Bootstrap Confidence Intervals
boot = ReshapedBootstrap(B=50, bandwidth=0.3, n_starts=3, seed=42)
boot_estimates = boot.fit_bootstrap(X, Y, beta_hat)
ci_95 = boot.confidence_intervals(beta_hat, boot_estimates, alpha=0.05)
print("95% Confidence Intervals:\n", ci_95)
```

---

## API Reference

### `MaxScoreEstimator`

Computes Manski's Maximum Score Estimator with the first coefficient fixed to `1.0` ($\beta = [1, \gamma^T]$) to resolve scale and sign indeterminacy. Optimization is performed using multi-start Differential Evolution (`scipy.optimize.differential_evolution`).

#### Initialization Arguments
```python
MaxScoreEstimator(bounds=(-10.0, 10.0), n_starts=3, seed=None)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `bounds` | `tuple[float, float]` | `(-10.0, 10.0)` | Lower and upper search bounds `(min, max)` applied to each free parameter $\gamma_j$ ($j = 2, \dots, d$). |
| `n_starts` | `int` | `3` | Number of independent Differential Evolution runs per optimization. The solution achieving the highest score (lowest negative objective value) is selected. |
| `seed` | `int` or `None` | `None` | Random seed used to initialize the internal RNG (`np.random.default_rng`) for reproducible multi-start runs. |

#### Methods
* **`fit(X, Y)`**
  * **`X`** (`np.ndarray`, shape `(n, d)`): Design matrix of covariates ($d \ge 2$). The first column corresponds to the normalized coefficient $\beta_1 = 1.0$.
  * **`Y`** (`np.ndarray`, shape `(n,)`): Binary outcome vector taking values `0` and `1` (or `0.0` and `1.0`).
  * **Returns:** `np.ndarray` of shape `(d,)` containing `[1.0, gamma_hat_1, ..., gamma_hat_{d-1}]`.

---

### `ReshapedBootstrap`

Implements the Cattaneo, Jansson, and Nagasawa (2020) Reshaped Bootstrap for cube-root ($n^{1/3}$) consistent Maximum Score inference. In each bootstrap replication, it maximizes the reshaped objective function incorporating the analytical plug-in Hessian quadratic centering term:
$$\tilde{M}_n^*(\gamma) = M_n^*(\gamma) - M_n(\gamma) + \frac{1}{2} (\gamma - \hat{\gamma})^T \tilde{H} (\gamma - \hat{\gamma})$$

#### Initialization Arguments
```python
ReshapedBootstrap(B=200, bandwidth=2.0, bounds=(-10.0, 10.0), n_starts=3, seed=None)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `B` | `int` | `200` | Number of bootstrap replications. Recommended minimum is `B >= 50` (typically `100–200`) for stable percentile confidence intervals. |
| `bandwidth` | `float` | `2.0` | Unified Gaussian kernel bandwidth applied to both the conditional probability regression $\hat{\kappa}(x) = \hat{E}[Y \mid X=x]$ and the analytical plug-in Hessian matrix $\tilde{H}$. Should be scaled to match the support of `X` (e.g., `0.3` for $X \sim U[-1, 1]$). |
| `bounds` | `tuple[float, float]` | `(-10.0, 10.0)` | Lower and upper search bounds `(min, max)` applied to each free parameter $\gamma_j$ ($j = 2, \dots, d$). |
| `n_starts` | `int` | `3` | Number of independent Differential Evolution runs executed within each bootstrap replication, retaining the best optimum. |
| `seed` | `int` or `None` | `None` | Random seed for reproducible resampling of $(X^*, Y^*)$ and Differential Evolution initializations. |

#### Methods
* **`fit_bootstrap(X, Y, beta_hat)`**
  * **`X`** (`np.ndarray`, shape `(n, d)`): Original covariate matrix.
  * **`Y`** (`np.ndarray`, shape `(n,)`): Original binary outcome vector (`0` and `1`).
  * **`beta_hat`** (`np.ndarray`, shape `(d,)`): Point estimate returned by `MaxScoreEstimator.fit(X, Y)` with `beta_hat[0] == 1.0`.
  * **Returns:** `np.ndarray` of shape `(B, d)` containing the bootstrap parameter estimates $\beta^{*(b)}$.
* **`confidence_intervals(beta_hat, bootstrap_estimates, alpha=0.05)`**
  * **`beta_hat`** (`np.ndarray`, shape `(d,)`): Original point estimate vector.
  * **`bootstrap_estimates`** (`np.ndarray`, shape `(B, d)`): Matrix of bootstrap draws returned by `fit_bootstrap`.
  * **`alpha`** (`float`, default `0.05`): Significance level for two-sided $(1 - \alpha)$ percentile confidence intervals (e.g., `0.05` produces 2.5% and 97.5% bounds).
  * **Returns:** `np.ndarray` of shape `(d, 2)` where column `0` is the lower bound and column `1` is the upper bound.

---

### `PluginHessianEstimator`

Computes the analytical Gaussian kernel derivative plug-in estimator for the Hessian matrix $\tilde{H}$ with respect to the free parameters $\gamma = (\beta_2, \dots, \beta_d)^T$, matching `Hhat.cpp` from Cattaneo et al. (2020).

#### Initialization Arguments
```python
PluginHessianEstimator(bw=2.0)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `bw` | `float` | `2.0` | Smoothing bandwidth for the standard Gaussian kernel derivative $K'(u) = -\frac{u}{\sqrt{2\pi}} e^{-u^2/2}$. |

#### Methods
* **`estimate(beta_hat, X, Y)`**
  * **`beta_hat`** (`np.ndarray`, shape `(d,)`): Point estimate vector.
  * **`X`** (`np.ndarray`, shape `(n, d)`): Covariate matrix.
  * **`Y`** (`np.ndarray`, shape `(n,)`): Binary outcome vector (`0` and `1`).
  * **Returns:** Symmetric `np.ndarray` of shape `(d - 1, d - 1)` representing the estimated Hessian matrix $\tilde{H}$.

---

## Reference
* Cattaneo, M. D., Jansson, M., & Nagasawa, K. (2020). *Bootstrap based inference for cube root asymptoticss*. Econometrica, 88(5), 2203–2219.