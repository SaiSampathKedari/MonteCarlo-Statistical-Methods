import numpy as np
from mcmc.utils.mcmc_utils import *
from mcmc.distributions.gaussian import *

def logpdf_banana_eval(x: np.ndarray):
    u = np.array([x[0], x[1] + x[0]**2 + 1])
    mean = np.zeros(2)
    cov = build_2D_covariance_matrix(1.0,1.0,0.9)
    return logpdf_multivariate_normal_eval(u, mean, cov)
    
def pdf_banana_eval(x: np.ndarray):
    return np.exp(logpdf_banana_eval(x))