import numpy as np
from mcmc.utils.mcmc_utils import *

def logpdf_multivariate_normal_eval(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    x = point in d dimention # shape (d,)
    mean = mean of the multivariate normal distribution # shape (d,)
    cov = covariance of the multivariate normal distribution # shape (d,d)
    
    outpu = logpdf evauation of point x
    """
    d = x.shape[0]
    
    # log normalization constant:  -(d/2)*log(2π) - (1/2)*log|Σ|
    log_preexp = -0.5 * (d * np.log(2 * np.pi) + np.log(np.linalg.det(cov)))
    
    # difference
    diff = x - mean  # (d, 1)
    
    # Compute con^{-1}(x - mean)
    sol = np.linalg.solve(cov, diff)     # (d, )
    
    inexp = -0.5*np.dot(diff, sol)
    
    # Final log-pdf 
    return log_preexp + inexp

def pdf_multivariate_normal_eval(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    x = point in d dimention # shape (d,)
    mean = mean of the multivariate normal distribution # shape (d,)
    cov = covariance of the multivariate normal distribution # shape (d,d)
    
    output = pdf evauation of point x
    """
    d = x.shape[0]
    
    # Normalization constant for MVN
    preexp = 1.0 / ((2 * np.pi)**(d / 2) * np.linalg.det(cov)**0.5)
    
    # difference
    diff = x - mean  # (d, 1)
    
    # Compute con^{-1}(x - mean)
    sol = np.linalg.solve(cov, diff)     # (d, )
    
    inexp = -0.5*np.dot(diff, sol)
    
    # Final PDF 
    return preexp * np.exp(inexp)