import numpy as np
from typing import Callable

def sample_exponential(lambda_param: float, n: int) -> np.ndarray:
    uniform_samples = np.random.rand(n)
    return (-1* lambda_param*np.log(uniform_samples))