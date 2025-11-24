from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.stats as stats

from mc_core.utils.montecarlo import *


@dataclass
class ControlVariateSamples:
    """Container for all outputs of the control variate estimator."""
    mc_g: MonteCarloEstimate
    mc_h: MonteCarloEstimate
    estimate_cv: float
    weight: float
    cov: np.ndarray
    correlation: float
    


def control_variate(
    n_num_samples: int,
    proposal_sample_generator: Callable[[int], np.ndarray],
    g_evaluator: Callable[[np.ndarray], np.ndarray],
    h_evaluator: Callable[[np.ndarray], np.ndarray],
    h_expected: Optional[np.ndarray] = None,
    cov_mat_g_h: Optional[np.ndarray] = None,
    m_num_samples: Optional[int] = None,
) -> ControlVariateSamples:
    
    """
    General Control Variate Estimator.

    Parameters
    ----------
    n_num_samples : int
        Number of samples for computing g and h over SHARED samples.
    proposal_sample_generator : Callable
        Function that generates samples X ~ p(x). 
    g_evaluator : Callable
        Computes g(X).
    h_evaluator : Callable
        Computes h(X).
    h_expected : float (optional)
        Known E[h(X)]. If None, it will be estimated using m independent samples.
    cov_mat_g_h : ndarray (optional)
        Precomputed covariance matrix. Useful for experiments.
    m_num_samples : int (optional)
        Number of samples for approximate CV to estimate E[h].

    Returns
    -------
    ControlVariateSamples
        A dataclass containing everything.
    """
    
    # 1. Validation
    if h_expected is None and m_num_samples is None:
        raise ValueError(
            "You must provide either h_expected or m_num_samples."
        )
    
    # 2. Generate SHARED samples for both g and h (required for CV)
    samples = proposal_sample_generator(n_num_samples)
    
    # For monte_carlo() interface: wrap a constant sample-returning function
    def shared_sampler(n):
        return samples
    
    # 3. Monte Carlo estimates (shared samples!)
    mc_g = monte_carlo(n_num_samples, shared_sampler, g_evaluator, cumsum=False)
    mc_h = monte_carlo(n_num_samples, shared_sampler, h_evaluator, cumsum=False)
    
    # 4. Compute covariance matrix Cov(g(X), h(X))
    if cov_mat_g_h is None:
        cov_mat_g_h = np.cov(mc_g.evaluations, mc_h.evaluations, rowvar=True) # shape 2 x 2
    
    var_g  = cov_mat_g_h[0, 0]
    cov_gh = cov_mat_g_h[0, 1]
    var_h  = cov_mat_g_h[1, 1]
    
    # 5. Compute correlation (for diagnostics)
    correlation = cov_gh / np.sqrt( var_g * var_h)
    
    # 6. Compute optimal CV weight (alpha*)
    weight = - cov_gh / var_h
    
    # 7. Estimate E[h] if needed (approximate CV)
    if h_expected is None:
        h_expected = monte_carlo(m_num_samples, 
                                 proposal_sample_generator, 
                                 h_evaluator, 
                                 cumsum=False).estimate
    
    # 8. Final control variate estimator
    estimate_cv = mc_g.estimate + weight * ( mc_h.estimate - h_expected)
    
    # 9. Return everything in a ControlVariateSamples object
    return ControlVariateSamples(mc_g,
                                 mc_h,
                                 estimate_cv,
                                 weight,
                                 cov_mat_g_h,
                                 correlation)
    
    