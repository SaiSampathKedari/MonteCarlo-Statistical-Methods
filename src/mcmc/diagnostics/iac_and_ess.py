import numpy as np
from mcmc.diagnostics.autocorrelation import *

def integrated_autocorrelation(
    samples: np.ndarray,
    max_lag: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Integrated Autocorrelation Time (IAC) per dimension
    by summing autocorrelation values until the first negative correlation.


    Parameters
    ----------
    samples : np.ndarray
        MCMC samples of shape (N, d)

    max_lag : int
        Maximum lag (step=1 always for IAC)

    Returns
    -------
    iac : np.ndarray
        Integrated autocorrelation per dimension (d,)

    acf_values : np.ndarray
        Autocorrelation values (L, d)

    lag_values : np.ndarray
        Lag indices (L,)
    """

    # 1. compute ACF using YOUR function
    lag_values, acf_values = autocorrelation_vectorized(
        samples,
        max_lag=max_lag,
        step=1
    )

    L, d = acf_values.shape
    iac = np.ones(d)  # start with 1 from lag 0 term

    for k in range(d):
        r = acf_values[:, k]

        cutoff = L
        for lag in range(1, L):
            if r[lag] < 0:
                cutoff = lag
                break
        
        iac[k] = 1 + 2 * np.sum(r[1:cutoff])

    return iac, acf_values, lag_values

def effective_sample_size(
    samples: np.ndarray,
    iac: np.ndarray
) -> np.ndarray:
    """
    Compute Effective Sample Size (ESS) per dimension.

    Parameters
    ----------
    samples : np.ndarray
        MCMC samples of shape (N, d)

    iac : np.ndarray
        Integrated autocorrelation time per dimension (d,)

    Returns
    -------
    ess : np.ndarray
        Effective sample size per dimension (d,)
    """
    N = samples.shape[0]
    return N / iac
