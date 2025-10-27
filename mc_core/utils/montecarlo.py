"""Monte Carlo Utilities."""
from dataclasses import dataclass
import numpy as np
from typing import Callable

@dataclass
class MonteCarloEstimate:
    estimate: np.ndarray
    samples: np.ndarray
    evaluations: np.ndarray

def monte_carlo(num_samples: int,
                sample_generator: Callable[[int], np.ndarray],
                g_evaluator: Callable[[np.ndarray], np.ndarray],
                cumsum: bool= False):
    """
    Perform Monte Carlo sampling.

    Inputs
    ------
    num_samples: int
        number of samples
    sample_generator: Callable[[int], np.ndarray]
        A function that generates samples with signature sample_generator(nsamples)
    g_evaluator: Callable[[np.ndarray], np.ndarray] 
        Function that takes as inputs the samples and outputs the evaluations.
                 The outputs can be any dimension, however the first dimension should have size *num_samples*
    cumsum: bool, optional 
        An option to return estimators of all sample sizes up to num_samples

    Returns
    -------
    MonteCarloEstimate:
        A Monte Carlo estimator of the mean, samples, and evaluations
    """
    samples = sample_generator(num_samples)
    evaluations = g_evaluator(samples)
    if cumsum is False:
        estimate =  np.sum(evaluations, axis=0) / float(num_samples)
    else:
        estimate = np.cumsum(evaluations, axis=0) / np.arange(1, num_samples + 1, dtype=np.float64)

    return MonteCarloEstimate(estimate, samples, evaluations)