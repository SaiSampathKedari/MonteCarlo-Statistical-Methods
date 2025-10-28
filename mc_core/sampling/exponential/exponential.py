import numpy as np
from typing import Callable

from mc_core.utils.uniform import uniform_sampleGenerator


def exponential_sampleGenerator(lambda_parameter: float, n: int) -> np.ndarray:
    uniform_samples = np.random.rand(n)
    return (-1* lambda_parameter*np.log(uniform_samples))