from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from mcmc.algorithms.metropolis_hastings import *

@dataclass
class DRResult(MCMCResultBase):
    """
    Result object for the Delayed Rejection algorithm.
    Inherits the standard MCMC fields without modifiction.
    """
    pass

def DRA_l2_acceptance_prob(
    log_f_y2:   float,
    log_f_x :   float,
    a1_y2_y1:   float,
    a1_x_y1 :   float,
    x       :   float,
    y1      :   float,
    y2      :   float,
    proposal_logpdf: LogCondPDF,
    cov: np.ndarray,
    gamma   :   float
):
    eps = 1e-12  # small float
    
    log_q1_y1_y2 = proposal_logpdf(y1, y2, cov)
    log_q1_y1_x  = proposal_logpdf(y1, x, cov)
    
    log_q2_x_y2_y1 = proposal_logpdf(x, y2, gamma*cov)
    log_q2_y2_y1_x = proposal_logpdf(y2, y1, gamma*cov)
    
    log_numerator =   log_f_y2 \
                    + log_q1_y1_y2 \
                    + log_q2_x_y2_y1\
                    + np.log(max(1.0 - a1_y2_y1, eps)) 
                    
    log_denominator=  log_f_x \
                    + log_q1_y1_x \
                    + log_q2_y2_y1_x \
                    + np.log(max(1.0 - a1_x_y1, eps))
    
    inexp = log_numerator - log_denominator
    
    if inexp >= 0:
        return 1.0

    # acceptance probability
    a2_x_y1_y2 = np.exp(log_numerator - log_denominator)
    
    return a2_x_y1_y2

def dr_mcmc(
    initial_sample : np.ndarray,
    initial_cov: np.ndarray,
    num_samples: int,
    target_logpdf: LogPDF,
    proposal_logpdf: LogCondPDF,
    proposal_sampler: ProposalSampler,
    gamma: float = 0.5,
    verbose: bool=True) -> DRResult:
    
    """
    Delayed Rejection MCMC Algorithm
    """
    
    # dimenstion of the state
    d = initial_sample.shape[0]
    
    # allocate space for all samples: shape (num_samples, d)
    samples = np.zeros((num_samples, d))
    samples[0] = initial_sample
    
    # mask indicating which proposals were accepted
    acceptance_mask = np.zeros(num_samples, dtype=bool)
    
    # initial proposal covariance (scaled random-walk form)
    eps= 1e-3
    sd = 2.4**2/float(d)
    S0 = sd* initial_cov + sd* eps * np.eye(d)
    Sk = S0
    
    # current state x and log f(x)
    x = initial_sample
    log_f_x = target_logpdf(x)
    
    acceptance_count = 0
    for k in range(1, num_samples):        
        # 1)  Level 1 proposal: y1 ~ q(· | x)
        y1 = proposal_sampler(x, Sk)
        
        # 2) compute log f(y1)
        log_f_y1 = target_logpdf(y1)
        
        # 3) Compute acceptance probability a(x,y1)
        a1_x_y1 = mh_acceptance_prob(
            log_f_x, log_f_y1,
            x, y1,
            proposal_logpdf, Sk)
        
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
            y2 = proposal_sampler(x, gamma*Sk)
            log_f_y2 = target_logpdf(y2)
            
            # 6) compute a_1(y_2, y_1)
            a1_y2_y1 = mh_acceptance_prob(
                log_f_y2, log_f_y1,
                y2, y1,
                proposal_logpdf, Sk)
            
            # 7) compute level-2 acceptance a2(x, y1, y2)
            a2_x_y1_y2 = DRA_l2_acceptance_prob(
                log_f_y2, log_f_x,
                a1_y2_y1, a1_x_y1,
                x, y1, y2,
                proposal_logpdf, Sk, gamma )
            
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
        
        # progress indicator (optional)
        if verbose and (k % 1000 == 0):
            print(f"Finished sample {k}, acceptance ratio = {acceptance_count / k:.3f}")
        
    # Return
    return DRResult(
        samples=samples,
        accept_rate= acceptance_count/float(num_samples-1),
        accept_mask=acceptance_mask)