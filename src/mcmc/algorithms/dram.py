from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from mcmc.algorithms.metropolis_hastings import *
from mcmc.algorithms.delayed_rejection import *

@dataclass
class DRAMResult(MCMCResultBase):
    """
    Result object for the Delayed Rejection Adaptive Metropolis algorithm.
    Inherits the standard MCMC fields without modifiction.
    """
    pass

def dram_mcmc(
    initial_sample : np.ndarray,
    initial_cov: np.ndarray,
    num_samples: int,
    target_logpdf: LogPDF,
    proposal_logpdf: LogCondPDF,
    proposal_sampler: ProposalSampler,
    k0: int=50,
    freq_of_update: int=100,
    gamma: float = 0.5,
    verbose: bool=True) -> DRAMResult:
    
    """
    Delayed Rejection Adaptive Metropolis MCMC Algorithm
    """
    
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
    Sk = S0.copy()         # always-updated covariance
    Sk_used = Sk.copy()    # used for proposing (updated only at freq)
    
    # initial state
    x = initial_sample
    log_f_x = target_logpdf(x)
    
    # Previos storage
    meanPrev = x.copy()
    
    acceptance_count = 0
    for k in range(1, num_samples):       
        #  0) If k == k0, compute initial covariance from the first k samples
        if k == k0:
            meanPrev = np.mean(samples[:k], axis=0)
            CkPrev = np.cov(samples[:k], rowvar=False)
            Sk = sd* CkPrev + sd * eps * np.eye(d)
            
            # IMPORTANT: when hitting k0, Sk_used must be updated
            Sk_used = Sk.copy()
        
        #  1) Propose y1 ~ q(. | x, Sk)
        y1 = proposal_sampler(x, Sk_used)
        
        # 2) compute log f(y1)
        log_f_y1 = target_logpdf(y1)
        
        # 3) Compute acceptance probability a(x,y1)
        a1_x_y1 = mh_acceptance_prob(
            log_f_x, log_f_y1,
            x, y1,
            proposal_logpdf, Sk_used)
        
        # 4) Accept Proposed or go to level 2
        u = np.random.rand()
        if u < a1_x_y1:
            # Accept the Proposed Sample
            samples[k] = y1
            
            x = y1
            log_f_x = log_f_y1
            
            acceptance_count += 1
            acceptance_mask[k] = True
        else:
            # 5) Level 2 proposal: y2 ~ q2(· | x)
            y2 = proposal_sampler(x, gamma*Sk_used)
            log_f_y2 = target_logpdf(y2)

            # 6) compute a_1(y_2, y_1)
            a1_y2_y1 = mh_acceptance_prob(
                log_f_y2, log_f_y1,
                y2, y1,
                proposal_logpdf, Sk_used)
            
            # 7) compute level-2 acceptance a2(x, y1, y2)
            a2_x_y1_y2 = DRA_l2_acceptance_prob(
                log_f_y2, log_f_x,
                a1_y2_y1, a1_x_y1,
                x, y1, y2,
                proposal_logpdf, Sk_used, gamma )
            
            # 8) Accept proposed y2 or Reject and take x
            u2 = np.random.rand()
            if u2 < a2_x_y1_y2:
                # Accept the Proposed sample y2
                samples[k] = y2
                
                x = y2
                log_f_x = log_f_y2
                
                acceptance_count += 1
                acceptance_mask[k] = True
            else:
                # remain at x
                samples[k] = x
                acceptance_mask[k] = False
        
        # 9)  Update covariance S_k recursively (Roberts–Rosenthal)
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
            
            # Use new Sk only every freq_of_update iterations
            if k % freq_of_update == 0:
                Sk_used = Sk.copy()
        
        # Optional progress printing
        if verbose and (k % 1000 == 0):
            print(f"Finished sample {k}, acceptance ratio = {acceptance_count / k:.3f}")
    
    # 6) Return 
    return DRAMResult(
        samples=samples,
        accept_rate=acceptance_count/float(num_samples-1),
        accept_mask=acceptance_mask)       