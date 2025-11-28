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