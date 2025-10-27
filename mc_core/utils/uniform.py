import numpy as np
from typing import Callable

def uniform_sampleGenerator(n: int) -> np.ndarray:
    """Return n samples from the Uniform(0,1) distribution."""
    
    return np.random.rand(n)