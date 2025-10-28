import numpy as np
from typing import Callable

from mc_core.sampling.direct.exponential.exponential import *

def gamma_int_parameters_sampleGenerator(alpha_param: int, 
                                         beta_param: float, 
                                         n:int) -> np.ndarray:
    exponential_samples = exponential_sampleGenerator(beta_param, n*alpha_param).reshape(n, alpha_param)
    return np.sum(exponential_samples, axis=1)
    
