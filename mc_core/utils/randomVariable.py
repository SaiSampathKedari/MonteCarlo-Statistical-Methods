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