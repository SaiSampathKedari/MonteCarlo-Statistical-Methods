from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from mcmc.algorithms.metropolis_hastings import *


@dataclass
class AMResult(MCMCResultBase):
    """
    Result object for the Adaptive Metropolis algorithm.
    Inherits the standard MCMC fields without modifiction.
    """
    pass

def am_mcmc(
    initial_sample : np.ndarray,
    initial_cov: np.ndarray,
    num_samples: int,
    target_logpdf: LogPDF,
    proposal_logpdf: LogCondPDF,
    proposal_sampler: ProposalSampler,
    k0: int=50,
    freq_of_update: int=100,
    verbose: bool=False) -> AMResult:
    
    # dimension of the state
    d = initial_sample.shape[0]
    
    # allocate array for samples: shape (num_samples, d)
    samples = np.zeros((num_samples, d))
    samples[0] = initial_sample
    
    # acceptance mask
    acceptance_mask = np.zeros(num_samples, dtype=bool)
    
    # initial proposal covariance (scaled random walk)
    eps= 1e-3
    sd = 2.4**2/float(d)
    S0 = sd* initial_cov + sd* eps * np.eye(d)
    Sk = S0
    
    # initial state
    x = initial_sample
    log_f_x = target_logpdf(x)
    
    # Previos storage
    meanPrev = x.copy()
    
    acceptance_count = 0
    for k in range(1, num_samples):
        
        #  0) If we reach adaptation point, compute initial S_k
        if k == k0:
            meanPrev = np.mean(samples[:k], axis=0)
            CkPrev = np.cov(samples[:k], rowvar=False)
            Sk = sd* CkPrev + sd * eps * np.eye(d)
            
        #  1) Propose y ~ q(. | x, Sk)
        y = proposal_sampler(x, Sk)
        
        # 2) target logpdf at proposed point: log f(y)
        log_f_y = target_logpdf(y)
        
        # 3) Compute acceptance probability a(x,y)
        a_x_y = mh_acceptance_prob(
            log_f_x,
            log_f_y,
            x,
            y,
            proposal_logpdf,
            Sk)
        
        # 4) Accept Proposed or Reject and take current 
        u = np.random.rand()
        if u < a_x_y:
            # Accept the Proposed Sample
            samples[k] = y
            x = y
            log_f_x = log_f_y
            acceptance_mask[k] = True
            acceptance_count += 1
        else:
            # Reject and take current Sample
            samples[k] = x
            acceptance_mask[k] = False
        
        # 5) Adapt covariance S_k (Roberts-Rosenthal AM)
        if k >= k0:
            xk = samples[k]
            old_mean = meanPrev
            
            # update mean
            meanPrev = (xk + k*meanPrev)/float(k+1)
            
            # update covariance: S_k
            Sk = ((k-1)/float(k)) * Sk \
                + (sd/float(k)) *(
                    eps * np.eye(d) 
                    + k     *   np.outer(old_mean, old_mean)
                    -(k+1)  *   np.outer(meanPrev, meanPrev) 
                    + np.outer(xk, xk)
                )
        # Optional progress printing
        if verbose and (k % 1000 == 0):
            print(f"Finished sample {k}, acceptance ratio = {acceptance_count / k:.3f}")
        
    # 6) Return 
    return AMResult(
        samples=samples,
        accept_rate=acceptance_count/float(num_samples-1),
        accept_mask=acceptance_mask)