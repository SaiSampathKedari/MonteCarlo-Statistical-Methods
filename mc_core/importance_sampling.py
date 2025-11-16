from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, Union

import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as scistats

@dataclass
class ImportanceSamples:
    # Drawn samples from the proposal distribution
    x_samples: np.ndarray           # (n,) proposed samples, X_i ~ g(d)
    
    # Desities
    f_vals: np.ndarray              # f(X_i): target pdf evaluated at samples
    g_vals: np.ndarray              # g(X_i): proposal pdf evaluated at sampels
    
    # Raw and normalized weights
    w_raw: np.ndarray               # w_i = f(X_i)/g(X_i)
    w_norm: np.ndarray              # normalized weights w_i/ sum(w_i)
    
    h_vals: np.ndarray              # Optional function evaluations h(X_i)
    estimate_is: np.ndarray         # 1/n * sum_i h(X_i) w_i  (unnormalized IS)
    estimate_snis: np.ndarray       # sum_i h(X_i) w_i / sum_i w_i (self-normalized IS)
    
    def ess(self) -> float:
            """
            Compute the Effective Sample Size (ESS) for importance sampling.
            
            ESS_raw = (sum w_i)^2 / sum w_i^2
            ESS_norm = 1 / sum (w_i_norm)^2
            
            Both formulas are mathematically equivalent.
            """
            w = self.w_raw
            if np.sum(w) == 0:
                return 0.0
            
            # Raw ESS
            ess_raw = (np.sum(w) ** 2) / np.sum(w ** 2)
            return ess_raw
    
    def ess_normalized(self) -> float:
        """
        ESS computed from normalized weights.
        
        ESS_norm = 1 / sum( w_i_norm^2 )
        """
        w = self.w_norm
        return 1.0 / np.sum(w ** 2)
    
        
def importance_sampling(
    num_samples: int,
    proposal_sample_generator: Callable[[int], np.ndarray],
    proposal_pdf_evaluator: Callable[[np.ndarray], np.ndarray],
    target_pdf_evaluator: Callable[[np.ndarray], np.ndarray],
    h_evaluator: Callable[[np.ndarray], np.ndarray],
    cumsum: bool= False
    ) -> ImportanceSamples:
    """
    importance sampling alogirthm
    """
    
    assert num_samples > 0, "n must be positive"
    
    # 1) Sample from proposal
    x_samples = proposal_sample_generator(num_samples)
    
    # 2) Evaluate pdfs
    g_vals = proposal_pdf_evaluator(x_samples)
    f_vals = target_pdf_evaluator(x_samples)
    
    # 3) Evaluate h on the samples
    h_vals = h_evaluator(x_samples)
            
    # 4) Compute raw weights w = f/g with guard for g=0
    with np.errstate(divide="ignore", invalid="ignore"):
        w_raw = np.where(g_vals > 0.0, f_vals / g_vals, 0.0)

    # 5) h(X_i)*w(X_i)
    h_w = h_vals * w_raw
    
    # 6) Normalized Weights
    w_norm = w_raw/np.sum(w_raw, axis=0)
    
    if cumsum is False:
        # 7) Normalized Weights
        w_sum = np.sum(w_raw, axis=0)
        
        # 8) Estimates
        # Unnormalized IS (classical estimator, unbiased if finite variance)
        estimate_is = np.sum(h_w, axis=0)/float(num_samples)
        estimate_snis = np.sum(h_w, axis=0)/float(w_sum)   
    else:
        # 7) Normalized weights
        w_sum = np.cumsum(w_raw, axis=0, dtype=np.float64)
        
        # 8) Estimates
        # Unnormalized IS (classical estimator, unbiased if finite variance)
        estimate_is = np.cumsum(h_w, axis=0)/np.arange(1, num_samples+1, dtype=np.float64)
        estimate_snis = np.cumsum(h_w, axis=0)/w_sum
        
    return ImportanceSamples(x_samples,
                             f_vals,
                             g_vals,
                             w_raw,
                             w_norm,
                             h_vals,
                             estimate_is,
                             estimate_snis)