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

def sampling_distribution(nsamples: int, 
                          ntrails: int, 
                          sample_generator: Callable[[int], np.ndarray],
                          g_evaluator: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Generate samples for plotting Sampling Distribution of size nsamples
    
    Inputs
    -------
    nsamples: int
        Sampling distribution of sample mean of nsamples random variable
    ntrails: int
        number of trails of nsamples, to form the distribution
    sample_generator: Callable[[int], np.ndarray]
        A function that generates samples with signature sample_generator(nsamples)
    g_evaluator: Callable[[np.ndarray], np.ndarray] 
        Function that takes as inputs the samples and outputs the evaluations.
                 The outputs can be any dimension, however the first dimension should have size *num_samples*
    """
    
    Estimates = np.zeros((ntrails))
    for i in range(ntrails):
        mc = monte_carlo(nsamples, sample_generator, g_evaluator)
        Estimates[i] = mc.estimate
    return Estimates

def normalize_sampling_distribution(nsamples: int, 
                          ntrails: int, 
                          sample_generator: Callable[[int], np.ndarray],
                          g_evaluator: Callable[[np.ndarray], np.ndarray],
                          mean: float,
                          std_deviation: float) -> np.ndarray:
    """
    Generate normalized (CLT) sampling distribution of sample means.
    
    Applies the transformation:
        Z_n = sqrt(n) * (Estimates - mean) / std_deviation

    Inputs
    -------
    nsamples : int
        Number of samples in each Monte Carlo trial.
    ntrails : int
        Number of Monte Carlo trials.
    sample_generator : Callable[[int], np.ndarray]
        Function generating nsamples from the target distribution.
    g_evaluator : Callable[[np.ndarray], np.ndarray]
        Function that evaluates the samples (e.g., identity for sample mean).
    mean : float
        True mean (E[X]) of the underlying distribution.
    std_deviation : float
        True standard deviation (sqrt(Var[X])) of the underlying distribution.

    Returns
    -------
    normalized_estimates : np.ndarray
        Array of standardized estimates approximating N(0, 1) as n increases.
    """
    
    assert std_deviation > 0, "Standard deviation must be positive."
    
    # Step 1: Get sample means from Monte Carlo trials
    estimates = sampling_distribution(nsamples, ntrails, sample_generator, g_evaluator)
    
    # Step 2: Apply CLT normalization
    normalized_estimates = np.sqrt(nsamples)*(estimates - mean)/(std_deviation)
    return normalized_estimates



