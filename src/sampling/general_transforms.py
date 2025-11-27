import numpy as np
from typing import Callable

def sample_uniform(n: int) -> np.ndarray:
    return np.random.rand(n)

def sample_exponential(lambda_param: float, n: int) -> np.ndarray:
    """
    Generate 'n' random samples from an Exponential(λ) distribution 
    using the inverse transform method.

    Parameters
    ----------
    lambda_param : float
        Scale parameter (λ) of the exponential distribution.
    n : int
        Number of random samples to generate.

    Returns
    -------
    np.ndarray
        Array of shape (n,) containing samples X = -λ * ln(U),
        where U ~ Uniform(0,1).
    """
    uniform_samples = np.random.rand(n) # U ~ Unif(0,1)
    return (-1* lambda_param*np.log(uniform_samples)) # X ~ Exp(λ)

def sample_gamma_integer_shape(alpha: int, beta: float, n:int) -> np.ndarray:
    
    """
    Generate 'n' random samples from a Gamma(alpha, beta) distribution
    when alpha is int (integer shape), using the sum-of-exponentials method.

    Parameters
    ----------
    alpha : int
        Shape parameter, must be a positive integer.
    beta : float
        Scale parameter of the Gamma distribution.
    n : int
        Number of random samples to generate.

    Returns
    -------
    np.ndarray
        Array of shape (n,) containing Gamma(alpha, beta) samples,
        computed as the sum of alpha independent Exp(beta) random variables.
    """
    exponential_samples = sample_exponential(beta, n*alpha).reshape(n, alpha)
    return np.sum(exponential_samples, axis=1)

def sample_chisquare_even(p: int, n: int) -> np.ndarray:
    """
    Generate 'n' random samples from a Chi-square(p) distribution
    using Uniform(0,1) transformations. Works only when p is even.

    Parameters
    ----------
    p : int
        Degrees of freedom (must be an even integer).
    n : int
        Number of random samples to generate.

    Returns
    -------
    np.ndarray
        Array of shape (n,) containing Chi-square(p) samples,
        generated via Gamma(p/2, 2) with integer shape p/2.
    """
    assert p % 2 == 0, "Degrees of freedom 'p' must be an even integer."
    return sample_gamma_integer_shape(p // 2, 2, n)

def sample_beta_integer_param(alpha: int, beta: int, n: int) -> np.ndarray:
    """
    Generate 'n' random samples from Beta(alpha, beta) distribution
    using the Gamma-to-Beta transformation.
    
    Parameters
    ----------
    alpha : int
        Shape parameter
    beta : int
        Shape parameter
    n : int
        Number of random samples to generate.
    
    Returns
    -------
    np.ndarray
        Array of Beta(alpha, beta) samples.
    """
    X1 = sample_gamma_integer_shape(alpha, 1, n)
    X2 = sample_gamma_integer_shape(beta, 1, n)
    return X1/(X1+X2)

def sample_logistic(mu: float, beta: float, n: int) -> np.ndarray:
    U = np.random.rand(n)
    X = np.log(U/(1-U))
    return mu + beta * X

def sample_laplace_standard(n: int) -> np.ndarray:
    U = np.random.rand(n)
    Z = np.where(U < 0.5, np.log(2.0 * U), -1*np.log(2.0 * (1.0 - U)))
    return Z

def sample_laplace(mu: float, sigma: float, n:int) -> np.ndarray:
    assert sigma > 0, "sigma must be positive"
    Z = sample_laplace_standard(n)
    return mu + sigma * Z