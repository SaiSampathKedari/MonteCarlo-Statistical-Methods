import numpy as np
from typing import Callable, Optional, Tuple


from matplotlib import pyplot as plt


def build_2D_covariance_matrix(std1: float, std2: float, correlation_coeff: float) -> np.ndarray:
    """
    Construct a 2x2 covariance matrix for a bivariate normal distribution.

    Parameters
    ----------
    std1 : float
        Standard deviation of variable 1. Must be > 0.
    std2 : float
        Standard deviation of variable 2. Must be > 0.
    correlation_coeff : float
        Correlation coefficient p ∈ [-1, 1].

    Returns
    -------
    np.ndarray
        A (2, 2) covariance matrix:
            [[std1²,       p·std1·std2],
             [p·std1·std2, std2²     ]]
    """
    assert std1 > 0, "Standard deviation std1 must be > 0."
    assert std2 > 0, "Standard deviation std2 must be > 0."
    assert -1.0 <= correlation_coeff <= 1.0, "Correlation must be in [-1, 1]."
    
    cov_mat = np.array([[std1**2, std1*std2*correlation_coeff], [std1*std2*correlation_coeff, std2**2]])
    
    return cov_mat


def multivariate_normal_pdf_eval(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Evaluate the Multivariate Normal PDF for samples stored as rows.

    Parameters
    ----------
    x : np.ndarray
        Samples, shape (N, d). Each row is a d-dimensional point.
    mean : np.ndarray
        Mean vector of the distribution, shape (d,).
    cov : np.ndarray
        Covariance matrix, shape (d, d), symmetric and positive definite.

    Returns
    -------
    np.ndarray
        PDF values for each sample, shape (N,).
    """
    N, d = x.shape

    # Normalization constant for MVN
    preexp = 1.0 / ((2 * np.pi)**(d / 2) * np.linalg.det(cov)**0.5)

    # Center data (broadcasting handles row-wise subtraction)
    diff = x - mean                          # (N, d)

    # Σ⁻¹ (x - μ) for each sample, done efficiently
    sol = np.linalg.solve(cov, diff.T).T     # (N, d)

    # Mahalanobis distance for each row
    quad = np.einsum("ij,ij->i", diff, sol)  # (N,)

    # Final PDF values
    return preexp * np.exp(-0.5 * quad)

def multivariate_normal_logpdf_eval(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Evaluate the Multivariate Normal LOG-PDF for samples stored as rows.
    This is numerically stable and should be used inside MCMC or inference algorithms.

    Parameters
    ----------
    x : np.ndarray
        Samples, shape (N, d). Each row is a d-dimensional point.
    mean : np.ndarray
        Mean vector, shape (d,).
    cov : np.ndarray
        Covariance matrix, shape (d, d). Must be positive definite.

    Returns
    -------
    np.ndarray
        Log-pdf values for each sample, shape (N,).
    """
    N, d = x.shape

    # log normalization constant:  -(d/2)*log(2π) - (1/2)*log|Σ|
    log_preexp = -0.5 * (d * np.log(2 * np.pi) + np.log(np.linalg.det(cov)))

    # Centered samples
    diff = x - mean                           # (N, d)

    # Compute Σ⁻¹ (x - μ)
    sol = np.linalg.solve(cov, diff.T).T      # (N, d)

    # Mahalanobis distance
    quad = np.einsum("ij,ij->i", diff, sol)   # (N,)

    # Final log-pdf
    return log_preexp - 0.5 * quad

def eval_normpdf_on_grid(x: np.ndarray, y: np.ndarray,
                         mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray] :
    """
    Evaluate a 2D multivariate normal PDF on a grid defined by x and y.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-axis grid points, shape (Nx,).
    y : np.ndarray
        1D array of y-axis grid points, shape (Ny,).
    mean : np.ndarray
        Mean vector of the Gaussian, shape (2,).
    cov : np.ndarray
        Covariance matrix of the Gaussian, shape (2, 2).

    Returns
    -------
    XX : np.ndarray
        Grid of x-coordinates, shape (Ny, Nx).
    YY : np.ndarray
        Grid of y-coordinates, shape (Ny, Nx).
    evals : np.ndarray
        PDF evaluated at each grid point, shape (Ny, Nx).
    """
    # 1) Build grid matrices
    XX, YY = np.meshgrid(x, y)        # shapes (Ny, Nx)

    # 2) Convert into row-wise samples: shape (Ny*Nx, 2)
    pts = np.column_stack((XX.ravel(), YY.ravel()))

    # 3) Log-pdf evaluation using your MVN logpdf
    logpdf_vals = multivariate_normal_logpdf_eval(pts, mean, cov)   # shape (Ny*Nx,)

    # 4) Convert to pdf and reshape back to grid
    pdf_vals = np.exp(logpdf_vals).reshape(XX.shape)

    return XX, YY, pdf_vals

def univariate_normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Compute univariate Gaussian PDF for vector x.
    
    Parameters
    ----------
    x : np.ndarray
        Input points, shape (N,).
    mean : float
        Mean of the distribution.
    std : float
        Standard deviation (> 0).
    
    Returns
    -------
    np.ndarray
        PDF values evaluated at x, shape (N,).
    """
    return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)

def univariate_normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Compute univariate Gaussian PDF for vector x.
    
    Parameters
    ----------
    x : np.ndarray
        Input points, shape (N,).
    mean : float
        Mean of the distribution.
    std : float
        Standard deviation (> 0).
    
    Returns
    -------
    np.ndarray
        PDF values evaluated at x, shape (N,).
    """
    return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)


def plot_bivariate_gauss(x, y, mean, cov, figsize=(10, 9)):
    
    # --- Univariate Gaussian PDF ---
    def univariate_pdf(z, m, s):
        return (1.0 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((z - m) / s) ** 2)

    # --- Joint PDF ---
    XX, YY = np.meshgrid(x, y)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    
    diff = pts - mean
    sol = np.linalg.solve(cov, diff.T).T
    quad = np.einsum("ij,ij->i", diff, sol)
    norm_const = 1.0 / ((2*np.pi)**(cov.shape[0]/2) * np.linalg.det(cov)**0.5)
    joint_pdf = norm_const * np.exp(-0.5 * quad)
    joint_pdf = joint_pdf.reshape(XX.shape)

    mean1, mean2 = mean
    std1 = np.sqrt(cov[0,0])
    std2 = np.sqrt(cov[1,1])

    # --- Figure Layout ---
    fig, ax = plt.subplots(2, 2, figsize=figsize,
                           gridspec_kw={"width_ratios":[1,1],
                                        "height_ratios":[1,1]})

    # -----------------------------------------------------
    # TOP LEFT — marginal of θ1
    # -----------------------------------------------------
    ax[0][0].plot(x, univariate_pdf(x, mean1, std1), linewidth=2.5)
    ax[0][0].set_title("Marginal PDF of $\\theta_1$", fontsize=14, weight="bold")
    ax[0][0].set_ylabel("$f_{\\theta_1}$", fontsize=12)
    ax[0][0].tick_params(labelsize=11)
    ax[0][0].grid(False)

    # -----------------------------------------------------
    # BOTTOM LEFT — Joint PDF
    # -----------------------------------------------------
    c = ax[1][0].contourf(XX, YY, joint_pdf, 
                          levels=80, cmap="viridis")
    ax[1][0].contour(XX, YY, joint_pdf, 
                     levels=10, colors="black", linewidths=0.4)

    ax[1][0].set_title("Joint PDF of $(\\theta_1, \\theta_2)$",
                       fontsize=14, weight="bold")
    ax[1][0].set_xlabel("$\\theta_1$", fontsize=12)
    ax[1][0].set_ylabel("$\\theta_2$", fontsize=12)
    ax[1][0].tick_params(labelsize=11)
    ax[1][0].set_aspect("equal")
    fig.colorbar(c, ax=ax[1][0], shrink=0.85)

    # -----------------------------------------------------
    # BOTTOM RIGHT — marginal of θ2
    # -----------------------------------------------------
    ax[1][1].plot(univariate_pdf(y, mean2, std2), y, linewidth=2.5)
    ax[1][1].set_title("Marginal PDF of $\\theta_2$", fontsize=14, weight="bold")
    ax[1][1].set_xlabel("$f_{\\theta_2}$", fontsize=12)
    ax[1][1].tick_params(labelsize=11)
    ax[1][1].grid(False)

    # -----------------------------------------------------
    # TOP RIGHT — empty
    # -----------------------------------------------------
    ax[0][1].set_visible(False)

    # --- Tighter Layout ---
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    return fig, ax
