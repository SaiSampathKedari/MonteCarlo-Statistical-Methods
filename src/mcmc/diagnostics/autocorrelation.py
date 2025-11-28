import numpy as np
from typing import Callable, Optional, Tuple
import matplotlib.pyplot as plt


def autocorrelation(
    samples: np.ndarray,
    max_lag: int,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the per-dimension autocorrelation function (ACF) 
    of a multivariate Markov chain using explicit Python loops.

    Parameters
    ----------
    samples : np.ndarray
        Array of shape (N, d) containing N samples in d dimensions.
        Each row is a single sample, each column a dimension.
    
    max_lag : int
        Maximum lag (inclusive of 0, exclusive of max_lag) for which
        to compute the autocorrelation.
    
    step : int, optional (default=1)
        Step size for the lag grid. ACF is computed for lags:
        [0, step, 2*step, ..., max_lag - 1].

    Returns
    -------
    lag_values : np.ndarray
        Array of shape (L,) listing the lag values at which the 
        autocorrelation was computed.
    
    acf_values : np.ndarray
        Array of shape (L, d). 
        acf_values[j, k] is the autocorrelation at lag lag_values[j]
        for dimension k.

    Notes
    -----
    This function uses a double loop (over lags and samples) and 
    is slower for large N. The vectorized version is preferred.
    """
    
    num_samples, dim = samples.shape
    mean = np.mean(samples, axis=0)

    # denominator per dimension
    denominator = np.zeros(dim)
    for i in range(num_samples):
        denominator += (samples[i] - mean)**2

    lag_values = np.arange(0, max_lag, step)
    acf_values = np.zeros((len(lag_values), dim))

    for j, lag in enumerate(lag_values):
        num_pairs = num_samples - lag
        corr_sum = np.zeros(dim)

        for t in range(num_pairs):
            corr_sum += (samples[t] - mean)*(samples[t + lag] - mean)

        acf_values[j] = corr_sum / denominator

    return lag_values, acf_values



def autocorrelation_vectorized(
    samples: np.ndarray,
    max_lag: int,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the per-dimension autocorrelation function (ACF) of a 
    multivariate Markov chain using vectorized NumPy operations.
    
    This implementation is significantly faster than the loop-based
    version and should be preferred for moderate or large chains.

    Parameters
    ----------
    samples : np.ndarray
        Array of shape (N, d) containing N samples in d dimensions.
        Each row is a single sample, each column a dimension.
    
    max_lag : int
        Maximum lag (inclusive of 0, exclusive of max_lag) for which
        to compute the autocorrelation.
    
    step : int, optional (default=1)
        Step size for the lag grid. ACF is computed for lags:
        [0, step, 2*step, ..., max_lag - 1].

    Returns
    -------
    lag_values : np.ndarray
        Array of shape (L,) listing the lag values at which the 
        autocorrelation is computed.
    
    acf_values : np.ndarray
        Array of shape (L, d). 
        acf_values[j, k] is the autocorrelation at lag lag_values[j]
        for dimension k.

    Notes
    -----
    This version computes:
        ACF(lag) = sum_{t=0}^{N-lag-1} (x_t - mean)*(x_{t+lag} - mean)
                   ---------------------------------------------------
                        sum_{t=0}^{N-1} (x_t - mean)^2
    for each dimension independently.
    """
    
    num_samples, dim = samples.shape
    mean = np.mean(samples, axis=0)

    # center the samples
    centered = samples - mean  # shape (N, d)

    # denominator per dimension
    denominator = np.sum(centered**2, axis=0)  # shape (d,)

    lag_values = np.arange(0, max_lag, step)
    acf_values = np.zeros((len(lag_values), dim))

    # vectorized computation for each lag
    for j, lag in enumerate(lag_values):
        acf_values[j] = (
            np.sum(centered[:num_samples - lag] * centered[lag:], axis=0) 
            / denominator
        )

    return lag_values, acf_values
    

def plot_autocorrelation_2d(
    lag_values: np.ndarray,
    acf_values: np.ndarray,
    dim_names: Tuple[str, str] = ("dimension 1", "dimension 2"),
    figsize: Tuple[int, int] = (12, 4),
    marker: str = "o"
):
    """
    Visualize autocorrelation function (ACF) for a 2-dimensional chain.

    Parameters
    ----------
    lag_values : np.ndarray
        Lags at which ACF is evaluated. Shape (L,).

    acf_values : np.ndarray
        Autocorrelation values. Shape (L, 2).

    dim_names : tuple of str, optional
        Titles for the two dimensions plotted side by side.

    figsize : tuple, optional
        Size of the matplotlib figure.

    marker : str, optional
        Matplotlib marker to use for ACF points.

    Notes
    -----
    ACF curves are plotted side by side for direct comparison.
    """

    if acf_values.shape[1] != 2:
        raise ValueError("This visualization function is only for 2D samples.")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for i, ax in enumerate(axes):
        ax.plot(lag_values, acf_values[:, i], marker=marker, linestyle="-")
        ax.set_title(f"Autocorrelation: {dim_names[i]}", fontsize=14)
        ax.set_xlabel("Lag", fontsize=12)
        ax.set_ylabel("ACF", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", linewidth=1, alpha=0.8)
        ax.set_ylim(-0.1, 1.05)  # keep consistent scale

    plt.tight_layout()
    plt.show()
    
def plot_mixing_2d(
    samples: np.ndarray,
    dim_names: Tuple[str, str] = ("x1", "x2"),
    figsize: Tuple[int, int] = (12, 5),
    color: str = "black"
):
    """
    Plot the mixing (traceplot) of a 2-dimensional MCMC chain.

    Parameters
    ----------
    samples : np.ndarray
        Array of shape (N, 2) containing the Markov chain samples.
        samples[:, 0] corresponds to dimension 1,
        samples[:, 1] corresponds to dimension 2.
    
    dim_names : tuple of str, optional
        Names of the dimensions to use for axis labels.

    figsize : tuple, optional
        Figure size for the plot.

    color : str, optional
        Line color for the trace plot.
    """

    if samples.shape[1] != 2:
        raise ValueError("plot_mixing_2d only works for 2D samples.")

    N = samples.shape[0]

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axs[0].plot(samples[:, 0], color=color)
    axs[0].set_ylabel(dim_names[0], fontsize=14)
    axs[0].set_title("Mixing (Traceplot)", fontsize=16)
    axs[0].grid(True, linestyle="--", alpha=0.4)

    axs[1].plot(samples[:, 1], color=color)
    axs[1].set_ylabel(dim_names[1], fontsize=14)
    axs[1].set_xlabel("Sample Index", fontsize=14)
    axs[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
