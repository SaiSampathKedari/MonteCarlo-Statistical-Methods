import numpy as np
from typing import Callable, Optional, Tuple

from matplotlib import pyplot as plt
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde


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

def plot_bivariate_gauss(
    x: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    figsize: Tuple[int, int] = (10, 9)
):
    """
    Plot a clean, publication-quality visualization of a 2D Gaussian:
    - Top-left: marginal PDF of θ1
    - Bottom-left: joint PDF contour (filled + outlines)
    - Bottom-right: marginal PDF of θ2
    - Top-right: empty (can later be used for MH trace, autocorr, hist, etc.)

    This function is intentionally minimal — no MCMC, no scatter.
    It is the base template for future MH overlays.
    """

    # --- Joint PDF computed using reusable function ---
    XX, YY, joint_pdf = eval_normpdf_on_grid(x, y, mean, cov)

    # --- Marginal parameters ---
    mean1, mean2 = mean
    std1 = np.sqrt(cov[0, 0])
    std2 = np.sqrt(cov[1, 1])

    # --- Create figure with fixed layout ---
    fig, ax = plt.subplots(
        2, 2, figsize=figsize,
        gridspec_kw={
            "width_ratios": [1.2, 1],
            "height_ratios": [1, 1.2]
        }
    )

    # ================================================
    # TOP LEFT — Marginal PDF of θ1
    # ================================================
    ax[0][0].plot(x, univariate_normal_pdf(x, mean1, std1), linewidth=2)
    ax[0][0].set_title("Marginal PDF of $\\theta_1$", fontsize=14, weight="bold")
    ax[0][0].set_ylabel("$f_{\\theta_1}$", fontsize=12)
    ax[0][0].tick_params(labelsize=11)
    ax[0][0].grid(False)

    # ================================================
    # BOTTOM LEFT — Joint PDF (contour + filled)
    # ================================================
    c = ax[1][0].contourf(XX, YY, joint_pdf, levels=60, cmap="viridis")
    ax[1][0].contour(XX, YY, joint_pdf, levels=10,
                     colors="black", linewidths=0.4)

    ax[1][0].set_title("Joint PDF of $(\\theta_1, \\theta_2)$",
                       fontsize=14, weight="bold")
    ax[1][0].set_xlabel("$\\theta_1$", fontsize=12)
    ax[1][0].set_ylabel("$\\theta_2$", fontsize=12)
    ax[1][0].tick_params(labelsize=11)
    ax[1][0].set_aspect("equal")

    fig.colorbar(c, ax=ax[1][0], shrink=0.85)

    # ================================================
    # BOTTOM RIGHT — Marginal PDF of θ2
    # ================================================
    ax[1][1].plot(univariate_normal_pdf(y, mean2, std2), y, linewidth=2)
    ax[1][1].set_title("Marginal PDF of $\\theta_2$", fontsize=14, weight="bold")
    ax[1][1].set_xlabel("$f_{\\theta_2}$", fontsize=12)
    ax[1][1].tick_params(labelsize=11)
    ax[1][1].grid(False)

    # ================================================
    # TOP RIGHT — Reserved (trace, autocorr, hist, etc.)
    # ================================================
    ax[0][1].set_axis_off()

    # Spacing
    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    return fig, ax

def scatter_matrix_clean(
    fignum,
    samples_list,
    truths=None,
    labels=[r"$\theta_1$", r"$\theta_2$"],
    mins=None,
    maxs=None,
    nbins=50,
    hist_plot=True,
    gamma=0.35,
    figsize=(11, 11)
):
    """
    Clean & professional version of your professor's scatter_matrix.
    - Same layout
    - No KDE curves
    - Uses θ₁, θ₂ notation
    - Truth lines only if provided
    - Large figure, clean spacing
    - Publication-level typography
    """

    nchains = len(samples_list)
    dim = samples_list[0].shape[1]
    all_samples = np.vstack(samples_list)

    # -----------------------------------------------------
    # Compute bounds
    # -----------------------------------------------------
    if mins is None:
        mins = np.quantile(all_samples, 0.01, axis=0)
    if maxs is None:
        maxs = np.quantile(all_samples, 0.99, axis=0)

    pad = 0.12 * (maxs - mins)
    mins = mins - pad
    maxs = maxs + pad

    # -----------------------------------------------------
    # Figure layout
    # -----------------------------------------------------
    fig = plt.figure(fignum, figsize=figsize)
    gs = GridSpec(dim, dim, figure=fig)
    axs = np.empty((dim, dim), dtype=object)

    fig.suptitle(
        "Posterior Samples (Metropolis–Hastings MCMC)",
        fontsize=22,
        fontweight="bold",
        y=0.94
    )

    cmap2d = cm.get_cmap("viridis")

    # -----------------------------------------------------
    # Main plotting grid
    # -----------------------------------------------------
    for i in range(dim):
        for j in range(dim):

            ax = fig.add_subplot(gs[i, j])
            axs[i, j] = ax

            # -------------------------------------------------
            # Diagonal = marginal histograms
            # -------------------------------------------------
            if i == j:
                vals = all_samples[:, i]

                ax.hist(
                    vals,
                    bins=nbins,
                    density=True,
                    color="#9dbcd4",
                    edgecolor="black",
                    alpha=0.75,
                )

                if truths is not None:
                    ax.axvline(truths[i], color="red", linewidth=2)

                ax.set_ylabel("density", fontsize=12)
                ax.set_xlabel(labels[i], fontsize=14)

                ax.grid(alpha=0.2)
                ax.set_ylim(bottom=0)

            # -------------------------------------------------
            # Lower triangle = joint density / scatter plot
            # -------------------------------------------------
            elif i > j:
                x = all_samples[:, j]
                y = all_samples[:, i]

                if hist_plot:
                    ax.hist2d(
                        x,
                        y,
                        bins=nbins,
                        range=[[mins[j], maxs[j]], [mins[i], maxs[i]]],
                        density=True,
                        cmap=cmap2d,
                        norm=colors.PowerNorm(gamma),
                    )
                else:
                    ax.scatter(
                        x,
                        y,
                        s=8,
                        color="#1f77b4",
                        alpha=0.35,
                        edgecolors="none"
                    )

                if truths is not None:
                    ax.plot(truths[j], truths[i], "x", color="red", markersize=10, mew=3)

                if j == 0:
                    ax.set_ylabel(labels[i], fontsize=14)
                if i == dim - 1:
                    ax.set_xlabel(labels[j], fontsize=14)

                ax.grid(alpha=0.2)

            # -------------------------------------------------
            # Upper triangle is always blank
            # -------------------------------------------------
            else:
                ax.set_visible(False)

            # -------------------------------------------------
            # Bounds
            # -------------------------------------------------
            if i == j:
                ax.set_xlim(mins[i], maxs[i])
            else:
                if i > j:
                    ax.set_xlim(mins[j], maxs[j])
                    ax.set_ylim(mins[i], maxs[i])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axs, gs