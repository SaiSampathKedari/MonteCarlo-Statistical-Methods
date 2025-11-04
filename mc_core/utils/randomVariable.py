from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scistats

@dataclass # a python decorator that builds some machinery for data classes
class RVPlotInfo:
    """Information needed to plot the cdf and pdf of a random variable."""
    x: np.ndarray
    cdf:np.ndarray
    pdf:np.ndarray

def get_rv_plot_info(rv: scistats.rv_continuous,
                     N: int = 100,
                     ppm_thresh: float = 1e-2) ->RVPlotInfo:
    
    """Get an array of x, pdf, and cdf for a scipy stat random variable.

    Args:
        rvs: A dictionary containing random variables to plot. The keys are the names
        N: Number of points to plot
        ppm_thresh: threshold probability for plotting tails

    Notes:
        Makes use of the ppm, pdf, and cdf functions provided by scipy.stats
        continuous random variables
    """
    x = np.linspace(rv.ppf(ppm_thresh), rv.ppf(1.0 - ppm_thresh), N)
    pdf = rv.pdf(x)
    cdf = rv.cdf(x)

    return RVPlotInfo(x, cdf, pdf)


def visualize(rvs: Dict[str, RVPlotInfo]) -> None:
    """Visualize the CDF and PDF of a random variable.

    Args:
        rvs: A dictionary containing random variables to plot. The keys are the names.

    """
    _, ax = plt.subplots(1, 2, figsize=(5, 3))
    for key, rv in rvs.items():
        ax[0].plot(rv.x, rv.cdf, '-', label=key)
        ax[1].plot(rv.x, rv.pdf, '-')
    ax[0].legend()
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$F_X(x)$')
    ax[0].set_title('CDF')
    ax[0].grid(which='both')

    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$f_X(x)$')
    ax[1].set_title('PDF')
    ax[1].grid(which='both')
    plt.tight_layout()
    
def visualize_samples(samples: np.ndarray, dist_name: str = "Distribution") -> None:
    """
    Visualize normalized histogram ( empirical PDF) of samples.
    
    Args:
        samples     : np.ndarray
            Random samples from any distribution
        dist_name   : str
            Name of the Distribution (for Title)
    """
    
    n = len(samples)
    
    # --- Adaptive bin width using Freedman-Diaconis rule ---
    q75, q25 = np.percentile(samples, [75,25])
    iqr = q75 - q25
    bin_width = 2* iqr/np.cbrt(n)
    if bin_width == 0:
        bin_width = (samples.max() - samples.min())/50 # fallback
    bins = int((samples.max() - samples.min()) / bin_width)
    
    # --- x-limits: focus on 0.1%–99.9% quantiles (ignore rare outliers) ---
    xmin, xmax = np.percentile(samples, [0.1, 99.9])
    
    # --- Plot normalized histogram ---
    plt.hist(samples, bins=bins, density=True, edgecolor="blue", alpha=0.7)
    plt.xlim(xmin, xmax)
    plt.title(f"{dist_name} Samples (Empirical PDF)", fontsize=14)
    plt.xlabel(r"$x$")
    plt.ylabel("Probability Density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_empirical_and_theoretical_pdf(
    samples: np.ndarray,
    rv: scistats.rv_continuous,
    dist_name: str = "Distribution",
    N: int = 300,
    ppm_thresh: float = 1e-3,
    save_path: str = None
) -> None:
    """
    Plot empirical normalized histogram of samples along with the 
    theoretical PDF of a scipy.stats continuous random variable
    
    Args:
        samples     : np.ndarray
            Random samples from any distribution
        rv : (scistats.rv_continuous): 
            scipy.stats distribution instance
            (e.g., scistats.gamma(a, scale)).
        dist_name   : str
            Name of the Distribution (for Title)
        N (int): Number of points for theoretical PDF plot.
        ppm_thresh (float): Tail probability cutoff for x-range in theoretical curve.
    """
    samples = np.asarray(samples)
    n = len(samples)

    # Theoretical x–pdf range
    x = np.linspace(rv.ppf(ppm_thresh), rv.ppf(1 - ppm_thresh), N)
    pdf = rv.pdf(x)

    # Freedman–Diaconis binning
    q75, q25 = np.percentile(samples, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / np.cbrt(n)
    if bin_width <= 0 or not np.isfinite(bin_width):
        bin_width = (samples.max() - samples.min()) / 50
    bins = max(25, int((samples.max() - samples.min()) / bin_width))

    # --- Styling ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Empirical histogram
    ax.hist(samples,
            bins=bins,
            density=True,
            color="#2980B9",      # deep professional blue
            alpha=0.55,
            edgecolor="none",
            label="Empirical PDF")

    # Theoretical PDF curve
    ax.plot(x, pdf,
            color="#F39C12",      # warm golden-orange contrast
            lw=2.5,
            label="Theoretical PDF")

    # Titles and labels
    ax.set_title(f"{dist_name}: Empirical vs Theoretical PDF",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)

    # Legend and grid
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)

    # Subtle axes and background tweaks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    
    # --- Save if path provided ---
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()
