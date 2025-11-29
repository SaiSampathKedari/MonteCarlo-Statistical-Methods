import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import replace


from mcmc.algorithms.metropolis_hastings import *


def mcmc_burnin_and_thin(result, burnin_frac: float, thin_frac: float):

    samples = result.samples
    mask = result.accept_mask
    N = samples.shape[0]

    start = int(burnin_frac * N)
    step = max(1, int(np.floor(1.0 / thin_frac)))

    indices = np.arange(start, N, step, dtype=int)

    new_samples = samples[indices]
    new_mask    = mask[indices]

    # acceptance rate
    if burnin_frac == 0.0:
        new_rate = np.mean(new_mask[1:]) if len(new_mask) > 1 else 0.0
    else:
        new_rate = np.mean(new_mask) if len(new_mask) > 0 else 0.0

    # the key line — preserves ALL other fields
    return replace(
        result,
        samples=new_samples,
        accept_rate=new_rate,
        accept_mask=new_mask
    )
    
def proposal_mvn_sampler(x: np.ndarray, cov: np.ndarray = None)->np.ndarray:
    """
    Multivariate normal random-walk proposal:
        y = x + L @ z
    where z ~ N(0, I) and L is the Cholesky factor of cov.
    """
    if cov is None:
        cov = np.identity(x.shape[0])
    L = np.linalg.cholesky(cov) # cov = LL^T
    z = np.random.randn(x.shape[0]) # standard normal vector
    sample = x + np.matmul(L,z)
    return sample

def proposal_mvn_logpdf_eval(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    x = point in d dimention # shape (d,)
    mean = mean of the multivariate normal distribution # shape (d,)
    cov = covariance of the multivariate normal distribution # shape (d,d)
    
    outpu = logpdf evauation of point x
    """
    d = x.shape[0]
    
    # log normalization constant:  -(d/2)*log(2π) - (1/2)*log|Σ|
    log_preexp = -0.5 * (d * np.log(2 * np.pi) + np.log(np.linalg.det(cov)))
    
    # difference
    diff = x - mean  # (d, 1)
    
    # Compute con^{-1}(x - mean)
    sol = np.linalg.solve(cov, diff)     # (d, )
    
    inexp = -0.5*np.dot(diff, sol)
    
    # Final log-pdf 
    return log_preexp + inexp