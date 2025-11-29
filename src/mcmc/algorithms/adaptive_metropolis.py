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
    freq_of_update: int=100) -> AMResult:
    
    # dimension of the state
    d = initial_sample.shape[0]
    
    # Computing Covariance
    eps= 0.001
    sd = 2.4**2/float(d)
    S0 = sd* initial_cov + sd* eps * np.eye(d)
    Sk = S0
    
    # allocate array for samples: shape (num_samples, d)
    samples = np.zeros((num_samples, d))
    samples[0] = initial_sample # store initial sample
    
    # acceptance mask: shape (num_samples, d)
    acceptance_mask = np.zeros(num_samples, dtype=bool) 
    
    # Previos storage
    meanPrev = samples[0]
    
    acceptance_count = 0
    current_sample = initial_sample
    for k in range(1, num_samples):
        
        # Initial Covariance Compute
        if k == k0:
            meanPrev = np.mean(samples[:k], axis=0)
            CkPrev = np.cov(samples[:k], rowvar=False)
            Sk = sd* CkPrev + sd * eps * np.eye(d)
            
        # 1) Propose a sample y ~ q(.|x, Sk)
        proposed_sample = proposal_sampler(current_sample, Sk)
        
        # 2) compute log f(x) and log f(y)
        current_target_logpdf_value = target_logpdf(current_sample)
        proposed_target_logpdf_value = target_logpdf(proposed_sample)
        
        # 3) Compute acceptance probability a(x,y)
        acceptance_prob = mh_acceptance_prob(
            current_target_logpdf_value,
            proposed_target_logpdf_value,
            current_sample,
            proposed_sample,
            proposal_logpdf,
            Sk)
        
        # 4) Accept Proposed or Reject and take current 
        u = np.random.rand()
        if u < acceptance_prob:
            # Accept the Proposed Sample
            samples[k] = proposed_sample
            acceptance_count += 1
            acceptance_mask[k] = True
        else:
            # Reject and take current Sample
            samples[k] = current_sample
            acceptance_mask[k] = False
        
        current_sample = samples[k]
        # 5) update Covariance and mean
        if k >= k0:
            xk = samples[k]
            meanPrev_old = meanPrev
            
            # update mean
            meanPrev = (xk + k*meanPrev)/float(k+1)
            
            # update covariance: S_k
            Sk = ((k-1)/float(k)) * Sk \
                + (sd/float(k)) *(
                    eps * np.eye(d) 
                    + k     *   np.outer(meanPrev_old, meanPrev_old)
                    -(k+1)  *   np.outer(meanPrev, meanPrev) 
                    + np.outer(xk, xk)
                )
    
    # 6) Return 
    return AMResult(
        samples=samples,
        accept_rate=acceptance_count/float(num_samples-1),
        accept_mask=acceptance_mask)