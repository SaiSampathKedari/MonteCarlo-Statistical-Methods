import numpy as np
from typing import Callable

from mc_core.sampling.exponential import *

def sample_gamma_int(alpha: int, beta: float, n:int) -> np.ndarray:
    exponential_samples = sample_exponential(beta, n*alpha).reshape(n, alpha)
    return np.sum(exponential_samples, axis=1)