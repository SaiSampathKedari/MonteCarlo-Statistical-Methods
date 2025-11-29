from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.stats as stats

from utils.montecarlo import *


LogPDF = Callable[[np.ndarray], float] # log f(x)
LogCondPDF = Callable[[np.ndarray, np.ndarray, np.ndarray], float] # log q(y|x) 
ProposalSampler = Callable[[np.ndarray, np.ndarray], np.ndarray]  # sample y ~ q(· | x)

@dataclass
class MCMCResultBase:
    """
    Base container for MCMC output.
    
    Attributes
    ----------
    samples : np.ndarray
        Array of generated samples with shape (N, d).
    accept_rate : float
        Overall acceptance rate of the chain.
    accept_mask : np.ndarray
        Boolean mask of length N indicating accepted proposals.
    """
    samples: np.ndarray
    accept_rate: float
    accept_mask: np.ndarray


@dataclass
class MHResult(MCMCResultBase):
    """
    Result object for the Metropolis-Hastings algorithm.
    Inherits the standard MCMC fields without modification.
    """
    pass


def mh_acceptance_prob(
    log_f_x : float,
    log_f_y : float,
    x: np.ndarray,
    y: np.ndarray,
    proposal_logpdf: LogCondPDF,
    cov: np.ndarray) -> float:

    log_q_y_x = proposal_logpdf(y, x, cov)
    log_q_x_y = proposal_logpdf(x, y, cov)
    log_a = (log_f_y + log_q_x_y) - (log_f_x + log_q_y_x)
    
    if log_a >= 0:
        return 1.0
    return np.exp(log_a)


def mh_mcmc(
    initial_sample: np.ndarray,
    initial_cov: np.ndarray,
    num_samples: int,
    target_logpdf: LogPDF,
    proposal_logpdf: LogCondPDF,
    proposal_sampler: ProposalSampler,
    verbose: bool = False) -> MHResult:

    """
    Metropolis-Hastings sampler for a target distribution f(x).

    Parameters
    ----------
    initial_sample : (d,) array
        Starting point of the chain.
    initial_cov : (d,d) array
        Base covariance for the random-walk proposal.
    num_samples : int
        Total number of MCMC iterations.
    target_logpdf : Callable
        Computes log f(x).
    proposal_logpdf : Callable
        Computes log q(y | x, Σ).
    proposal_sampler : Callable
        Draws y ~ q(· | x, Σ).
    verbose : bool
        Print progress every 1000 iterations.

    Returns
    -------
    MHResult
        samples : (N,d) array of states
        accept_rate : float
        accept_mask : (N,) boolean array
    """
    # dimension of the state
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
        # 1) Propose y ~ q(· | x, Sk)
        y = proposal_sampler(x, Sk)
        
        # 2) Evaluate target at proposed point: log f(y)
        log_f_y = target_logpdf(y)
        
        # 3) Compute MH acceptance probability a(x,y)
        a_x_y = mh_acceptance_prob(
            log_f_x,
            log_f_y,
            x,
            y,
            proposal_logpdf,
            Sk)
        
        # 4) Accept or reject the proposal
        u = np.random.rand()
        if u < a_x_y:
            # accept → move to y
            samples[k] = y
            x = y
            log_f_x = log_f_y
            acceptance_mask[k] = True
            acceptance_count += 1
        else:
            # reject → remain at x
            samples[k] = x
            acceptance_mask[k] = False
        
        # progress indicator (optional)
        if verbose and (k % 1000 == 0):
            print(f"Finished sample {k}, acceptance ratio = {acceptance_count / k:.3f}")
    
    # return full chain statistics
    return MHResult(
            samples=samples,
            accept_rate= acceptance_count/float(num_samples-1),
            accept_mask=acceptance_mask)
        