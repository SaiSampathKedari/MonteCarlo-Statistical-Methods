from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
import scipy.stats as scistats
import os
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

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

def animate_histogram_falling_sand_density(
    samples: np.ndarray,
    rv,
    dist_name: str = "Distribution",
    out_path: str = "notebooks/images/ch02_general_transformation/empirical_falling_sand_density.gif",
    N: int = 300,
    ppm_thresh: float = 1e-3,
    frames: int = 300,
    warmup_frames: int = 40,
    interval_ms: int = 40,
    fps: int = 25,
    figsize=(8, 5),
    facecolor: str = "white",
    sand_cmap: str = "Blues",     # continuous colormap for sand
    curve_color: str = "#E67E22", # orange theoretical PDF
    alpha_hist: float = 0.9,
    n_particles: int = 200,
    kind: str = "gif"
) -> str:
    """
    'Falling sand' animation with density-based color shading.
    Simulates grains falling from above and accumulating into
    a histogram that darkens with density, converging to the
    theoretical PDF curve.
    """

    samples = np.asarray(samples)
    n_total = len(samples)
    if n_total < 10:
        raise ValueError("At least 10 samples required for animation.")

    # --- Theoretical PDF ---
    x = np.linspace(rv.ppf(ppm_thresh), rv.ppf(1 - ppm_thresh), N)
    pdf = rv.pdf(x)

    # --- Bin setup ---
    bins = max(60, int(np.sqrt(n_total)))
    hist_vals_final, bin_edges = np.histogram(samples, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ymax = max(pdf.max(), hist_vals_final.max()) * 1.3

    # --- Figure setup ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(facecolor)

    # Plot theoretical PDF
    ax.plot(x, pdf, color=curve_color, lw=2.5, label="Theoretical PDF", zorder=3)

    # Initialize histogram bars (empty)
    bar_width = bin_edges[1] - bin_edges[0]
    bars = ax.bar(
        bin_centers, np.zeros_like(hist_vals_final),
        width=bar_width, color="#A9CCE3", alpha=alpha_hist,
        edgecolor="none", zorder=2
    )

    # Falling particles (sand grains)
    rng = np.random.default_rng(42)
    sand_x = rng.choice(samples, n_particles)
    sand_y = np.ones(n_particles) * ymax
    scatter = ax.scatter(sand_x, sand_y, s=20, c="#5DADE2", alpha=0.8, zorder=4)

    # Color normalization for density-based shading
    cmap = get_cmap(sand_cmap)
    norm = Normalize(vmin=0, vmax=hist_vals_final.max())

    # Axes
    ax.set_xlim(samples.min(), samples.max())
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"$x$", fontsize=12, style="italic")
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(f"{dist_name}: Empirical → Theoretical", fontsize=14, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    # --- Helper easing function ---
    def frame_to_heights(f):
        if f < warmup_frames:
            return np.zeros_like(hist_vals_final)
        t = (f - warmup_frames) / (frames - warmup_frames)
        t = 3 * t**2 - 2 * t**3  # ease-in-out smoothness
        return hist_vals_final * np.clip(t, 0, 1)

    def update(frame):
        # Histogram fill
        heights = frame_to_heights(frame)
        for rect, h in zip(bars, heights):
            color = cmap(norm(h))
            rect.set_height(h)
            rect.set_color(color)

        # Falling sand particles
        nonlocal sand_y
        fall_speed = ymax / (frames / 1.5)
        sand_y = np.maximum(sand_y - fall_speed, np.random.uniform(0.05 * ymax, ymax, size=sand_y.shape))
        scatter.set_offsets(np.column_stack([sand_x, sand_y]))

        # Update title
        ax.set_title(f"{dist_name}: Empirical → Theoretical   ({int(frame / frames * n_total)} samples)",
                     fontsize=14, fontweight="bold", pad=10)
        return bars, scatter

    # Animation
    anim = FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False)

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if kind.lower() == "mp4":
        writer = FFMpegWriter(fps=fps)
        anim.save(out_path, writer=writer, dpi=150)
    elif kind.lower() in {"gif", "apng"}:
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer, dpi=150)
    else:
        raise ValueError("kind must be 'gif' or 'mp4'.")

    plt.close(fig)
    return out_path