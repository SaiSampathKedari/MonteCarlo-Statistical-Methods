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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, LinearSegmentedColormap
import os
from scipy import stats as scistats


def animate_histogram_convergence_professional(
    samples: np.ndarray,
    rv,
    dist_name: str = "Distribution",
    out_path: str = "animations/histogram_convergence.mp4",
    N: int = 500,
    ppm_thresh: float = 1e-3,
    frames: int = 400,
    fps: int = 30,
    figsize=(14, 7),
    kind: str = "mp4"
) -> str:
    """
    Professional visualization of empirical histogram convergence to theoretical PDF.
    
    Shows:
    - Smooth histogram build-up with intelligent density-based coloring
    - Real-time convergence statistics (KS test, moments)
    - Ghost outline of theoretical distribution
    - Progressive sample accumulation
    
    Perfect for demonstrating:
    - Law of Large Numbers
    - Empirical distribution convergence
    - Monte Carlo convergence behavior
    - CDF transformation validation
    
    Parameters
    ----------
    samples : np.ndarray
        Random samples from the distribution
    rv : scipy.stats distribution
        Theoretical distribution (must have pdf, mean, std methods)
    dist_name : str
        Display name for the distribution
    out_path : str
        Output file path (.mp4 recommended)
    N : int
        Number of points for theoretical PDF curve
    ppm_thresh : float
        Tail threshold for plotting range
    frames : int
        Number of animation frames
    fps : int
        Frames per second
    figsize : tuple
        Figure size (width, height)
    kind : str
        Output format ('mp4' recommended)
    
    Returns
    -------
    str
        Path to saved animation
    """
    
    samples = np.asarray(samples)
    n_total = len(samples)
    if n_total < 10:
        raise ValueError("At least 10 samples required for animation.")
    
    # ==================== PROFESSIONAL COLOR SCHEME ====================
    COLOR_BG = '#FFFFFF'
    COLOR_GRID = '#E0E0E0'
    COLOR_TEXT = '#212121'
    COLOR_THEORETICAL = '#1976D2'      # Professional blue
    COLOR_EMPIRICAL_LIGHT = '#E3F2FD'  # Very light blue
    COLOR_EMPIRICAL_DARK = '#0D47A1'   # Deep blue
    COLOR_PANEL_BG = '#FAFAFA'
    COLOR_PANEL_BORDER = '#BDBDBD'
    COLOR_CONVERGENCE = '#388E3C'      # Green for good convergence
    
    # ==================== DATA PREPARATION ====================
    # Theoretical PDF
    x_min = rv.ppf(ppm_thresh)
    x_max = rv.ppf(1 - ppm_thresh)
    x = np.linspace(x_min, x_max, N)
    pdf_theoretical = rv.pdf(x)
    
    # Theoretical moments
    mean_theoretical = rv.mean()
    std_theoretical = rv.std()
    
    # Bin setup - adaptive based on sample size
    n_bins = max(40, min(100, int(np.sqrt(n_total))))
    
    # Y-axis limit
    ymax = 1.25 * pdf_theoretical.max()
    
    # Progressive sample counts - smooth progression
    sample_indices = np.unique(np.logspace(1, np.log10(n_total), frames).astype(int))
    sample_indices = np.clip(sample_indices, 10, n_total)
    frames = len(sample_indices)
    
    # ==================== FIGURE SETUP ====================
    plt.style.use('default')
    fig = plt.figure(figsize=figsize, facecolor=COLOR_BG)
    
    # Create grid: main plot (left) + stats panel (right)
    gs = fig.add_gridspec(1, 20, hspace=0.0, wspace=0.4)
    ax_main = fig.add_subplot(gs[0, :15])
    ax_stats = fig.add_subplot(gs[0, 16:])
    
    # Style main axis
    ax_main.set_facecolor(COLOR_BG)
    ax_main.grid(True, alpha=0.3, color=COLOR_GRID, linewidth=0.8)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['left'].set_color(COLOR_TEXT)
    ax_main.spines['bottom'].set_color(COLOR_TEXT)
    ax_main.spines['left'].set_linewidth(1.2)
    ax_main.spines['bottom'].set_linewidth(1.2)
    ax_main.tick_params(colors=COLOR_TEXT, which='both', labelsize=11)
    
    # ==================== THEORETICAL DISTRIBUTION ====================
    # Main theoretical curve
    line_theoretical, = ax_main.plot(x, pdf_theoretical, 
                                     color=COLOR_THEORETICAL, lw=3, 
                                     label='Theoretical PDF',
                                     zorder=5, solid_capstyle='round')
    
    # Subtle shadow for depth
    ax_main.plot(x, pdf_theoretical, color=COLOR_THEORETICAL, 
                lw=4.5, alpha=0.15, zorder=4, solid_capstyle='round')
    
    # Light fill under curve
    ax_main.fill_between(x, 0, pdf_theoretical, alpha=0.08, 
                         color=COLOR_THEORETICAL, zorder=1)
    
    # ==================== HISTOGRAM SETUP ====================
    # Create custom colormap: light blue -> deep blue based on density
    colors_hist = [COLOR_EMPIRICAL_LIGHT, COLOR_EMPIRICAL_DARK]
    n_bins_cmap = 256
    cmap_hist = LinearSegmentedColormap.from_list('empirical', colors_hist, N=n_bins_cmap)
    
    # Initialize empty histogram
    bins = np.linspace(samples.min(), samples.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bar_width = bins[1] - bins[0]
    
    # Create bar container
    bars = ax_main.bar(bin_centers, np.zeros(n_bins), 
                      width=bar_width, edgecolor='white', 
                      linewidth=0.5, alpha=0.9, zorder=3)
    
    # ==================== LABELS ====================
    ax_main.set_xlim(samples.min() - 0.05 * (samples.max() - samples.min()),
                    samples.max() + 0.05 * (samples.max() - samples.min()))
    ax_main.set_ylim(0, ymax)
    ax_main.set_xlabel(r'$x$', fontsize=15, color=COLOR_TEXT, fontweight='600')
    ax_main.set_ylabel('Probability Density', fontsize=15, color=COLOR_TEXT, fontweight='600')
    ax_main.set_title(f'{dist_name}: Empirical Convergence to Theoretical PDF', 
                     fontsize=17, color=COLOR_TEXT, fontweight='bold', pad=15)
    
    legend = ax_main.legend(frameon=True, fontsize=11, loc='upper right',
                           facecolor='white', edgecolor=COLOR_PANEL_BORDER,
                           framealpha=0.95, shadow=True)
    legend.get_frame().set_linewidth(1.2)
    
    # ==================== STATISTICS PANEL ====================
    ax_stats.set_facecolor(COLOR_PANEL_BG)
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis('off')
    
    # Border
    border = Rectangle((0, 0), 1, 1, transform=ax_stats.transAxes,
                       fill=False, edgecolor=COLOR_PANEL_BORDER, 
                       linewidth=1.5, zorder=0)
    ax_stats.add_patch(border)
    
    # Title
    ax_stats.text(0.5, 0.95, 'Convergence', ha='center', va='top',
                 fontsize=16, fontweight='bold', color=COLOR_TEXT)
    ax_stats.plot([0.1, 0.9], [0.91, 0.91], color=COLOR_PANEL_BORDER, 
                 lw=1.5, solid_capstyle='round')
    
    # Text objects for dynamic updates
    text_samples = ax_stats.text(0.1, 0.84, '', fontsize=11, color=COLOR_TEXT,
                                va='top', fontweight='500')
    
    ax_stats.plot([0.1, 0.9], [0.78, 0.78], color=COLOR_PANEL_BORDER, 
                 lw=1, alpha=0.5)
    
    text_mean_label = ax_stats.text(0.1, 0.72, 'Mean:', fontsize=11, 
                                    color=COLOR_TEXT, va='top', fontweight='600')
    text_mean_emp = ax_stats.text(0.1, 0.66, '', fontsize=10.5, 
                                  color=COLOR_TEXT, va='top', family='monospace')
    text_mean_theo = ax_stats.text(0.1, 0.61, '', fontsize=10.5, 
                                   color=COLOR_THEORETICAL, va='top', family='monospace')
    
    text_std_label = ax_stats.text(0.1, 0.54, 'Std Dev:', fontsize=11,
                                   color=COLOR_TEXT, va='top', fontweight='600')
    text_std_emp = ax_stats.text(0.1, 0.48, '', fontsize=10.5,
                                 color=COLOR_TEXT, va='top', family='monospace')
    text_std_theo = ax_stats.text(0.1, 0.43, '', fontsize=10.5,
                                  color=COLOR_THEORETICAL, va='top', family='monospace')
    
    ax_stats.plot([0.1, 0.9], [0.37, 0.37], color=COLOR_PANEL_BORDER,
                 lw=1, alpha=0.5)
    
    text_ks_label = ax_stats.text(0.1, 0.31, 'KS Statistic:', fontsize=11,
                                  color=COLOR_TEXT, va='top', fontweight='600')
    text_ks_value = ax_stats.text(0.1, 0.25, '', fontsize=10.5,
                                  color=COLOR_TEXT, va='top', family='monospace')
    text_ks_status = ax_stats.text(0.1, 0.19, '', fontsize=10,
                                   color=COLOR_CONVERGENCE, va='top', 
                                   fontweight='bold', style='italic')
    
    # Progress bar
    bar_y = 0.08
    bar_height = 0.05
    bar_bg = Rectangle((0.1, bar_y), 0.8, bar_height,
                       facecolor='white', edgecolor=COLOR_PANEL_BORDER,
                       linewidth=1.5, zorder=1)
    ax_stats.add_patch(bar_bg)
    
    bar_fill = Rectangle((0.1, bar_y), 0.0, bar_height,
                         facecolor=COLOR_THEORETICAL, edgecolor='none', 
                         zorder=2, alpha=0.8)
    ax_stats.add_patch(bar_fill)
    
    text_progress = ax_stats.text(0.5, bar_y + bar_height + 0.01, 'Progress',
                                 ha='center', va='bottom', fontsize=10,
                                 color=COLOR_TEXT, fontweight='500')
    
    # ==================== ANIMATION UPDATE ====================
    def update(frame_idx):
        n_current = sample_indices[frame_idx]
        current_samples = samples[:n_current]
        
        # Compute histogram
        hist_vals, _ = np.histogram(current_samples, bins=bins, density=True)
        
        # Normalize for colormap
        norm = Normalize(vmin=0, vmax=pdf_theoretical.max())
        
        # Update bars with density-based coloring
        for bar, height in zip(bars, hist_vals):
            bar.set_height(height)
            # Color based on density - darker = higher density
            color = cmap_hist(norm(height))
            bar.set_color(color)
        
        # Update legend to show histogram
        if frame_idx == 0:
            bars[0].set_label('Empirical Histogram')
            ax_main.legend(frameon=True, fontsize=11, loc='upper right',
                          facecolor='white', edgecolor=COLOR_PANEL_BORDER,
                          framealpha=0.95, shadow=True)
        
        # Compute statistics
        mean_emp = np.mean(current_samples)
        std_emp = np.std(current_samples, ddof=1)
        
        # KS test
        ks_stat, ks_pval = scistats.kstest(current_samples, rv.cdf)
        
        # Update statistics panel
        text_samples.set_text(f'n = {n_current:,} / {n_total:,}')
        
        text_mean_emp.set_text(f'  Empirical: {mean_emp:.4f}')
        text_mean_theo.set_text(f'  Theoretical: {mean_theoretical:.4f}')
        
        text_std_emp.set_text(f'  Empirical: {std_emp:.4f}')
        text_std_theo.set_text(f'  Theoretical: {std_theoretical:.4f}')
        
        text_ks_value.set_text(f'  D = {ks_stat:.4f}')
        
        # Convergence status based on KS p-value
        if ks_pval > 0.05:
            status_text = '✓ Good fit'
            status_color = COLOR_CONVERGENCE
        elif ks_pval > 0.01:
            status_text = '~ Moderate'
            status_color = '#F57C00'  # Orange
        else:
            status_text = '⨯ Converging...'
            status_color = '#E64A19'  # Red
        
        text_ks_status.set_text(f'  {status_text}')
        text_ks_status.set_color(status_color)
        
        # Update progress bar
        progress = n_current / n_total
        bar_fill.set_width(0.8 * progress)
        text_progress.set_text(f'Progress: {progress:.1%}')
        
        return (bars, text_samples, text_mean_emp, text_mean_theo,
                text_std_emp, text_std_theo, text_ks_value, 
                text_ks_status, bar_fill, text_progress)
    
    # ==================== BUILD ANIMATION ====================
    anim = FuncAnimation(
        fig, update, frames=frames, interval=1000/fps,
        blit=False, repeat=False
    )
    
    # ==================== SAVE ====================
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    
    if kind.lower() == "mp4":
        writer = FFMpegWriter(fps=fps, bitrate=3000,
                             codec='libx264',
                             extra_args=['-pix_fmt', 'yuv420p'])
        anim.save(out_path, writer=writer, dpi=150)
    else:
        raise ValueError("This function is optimized for MP4 output. Use kind='mp4'")
    
    plt.close(fig)
    print(f"✓ Professional histogram convergence animation saved to: {out_path}")
    return out_path


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, FancyBboxPatch
import os
from scipy import stats as scistats


def animate_convergence_elegant(
    samples: np.ndarray,
    rv,
    dist_name: str = "Distribution",
    out_path: str = "animations/convergence_elegant.mp4",
    N: int = 500,
    ppm_thresh: float = 1e-3,
    frames: int = 400,
    fps: int = 30,
    figsize=(12, 7),
    kind: str = "mp4"
) -> str:
    """
    Elegant, minimal convergence visualization with integrated metrics ribbon.
    
    Clean single-view design that shows:
    - Beautiful histogram-PDF overlay with transparency
    - Integrated metrics ribbon at top showing live convergence
    - Progress indicator
    - Minimal, uncluttered aesthetics
    
    Perfect for websites and presentations - modern, clean, professional.
    
    Parameters
    ----------
    samples : np.ndarray
        Random samples from the distribution
    rv : scipy.stats distribution
        Theoretical distribution (must have pdf, cdf, mean, std methods)
    dist_name : str
        Display name for the distribution
    out_path : str
        Output file path (.mp4 recommended)
    N : int
        Number of points for theoretical PDF curve
    ppm_thresh : float
        Tail threshold for plotting range
    frames : int
        Number of animation frames
    fps : int
        Frames per second
    figsize : tuple
        Figure size (width, height)
    kind : str
        Output format ('mp4' recommended)
    
    Returns
    -------
    str
        Path to saved animation
    """
    
    samples = np.asarray(samples)
    n_total = len(samples)
    if n_total < 10:
        raise ValueError("At least 10 samples required for animation.")
    
    # ==================== ELEGANT COLOR SCHEME ====================
    COLOR_BG = '#FAFAFA'           # Very light gray (softer than pure white)
    COLOR_PLOT_BG = '#FFFFFF'       # Pure white for plot area
    COLOR_GRID = '#EEEEEE'
    COLOR_TEXT = '#37474F'          # Blue-gray text
    COLOR_TEXT_LIGHT = '#78909C'    # Lighter blue-gray
    COLOR_PDF = '#D32F2F'           # Deep red for PDF
    COLOR_HIST = '#1976D2'          # Material blue for histogram
    COLOR_HIST_LIGHT = '#BBDEFB'    # Very light blue
    COLOR_RIBBON_BG = '#ECEFF1'     # Light blue-gray ribbon
    COLOR_ACCENT = '#FF6F00'        # Amber accent
    COLOR_SUCCESS = '#388E3C'       # Green for good convergence
    
    # ==================== DATA PREPARATION ====================
    # Theoretical PDF
    x_min = rv.ppf(ppm_thresh)
    x_max = rv.ppf(1 - ppm_thresh)
    x = np.linspace(x_min, x_max, N)
    pdf_theoretical = rv.pdf(x)
    
    # Theoretical moments
    mean_theoretical = rv.mean()
    std_theoretical = rv.std()
    
    # Bin setup
    n_bins = max(50, min(80, int(np.sqrt(n_total))))
    bins = np.linspace(samples.min(), samples.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bar_width = bins[1] - bins[0]
    
    # Progressive sample counts
    sample_indices = np.unique(np.logspace(1, np.log10(n_total), frames).astype(int))
    sample_indices = np.clip(sample_indices, 10, n_total)
    frames = len(sample_indices)
    
    # ==================== FIGURE SETUP ====================
    plt.style.use('default')
    fig = plt.figure(figsize=figsize, facecolor=COLOR_BG)
    
    # Main plot area with padding for ribbon
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.75])  # [left, bottom, width, height]
    
    ax.set_facecolor(COLOR_PLOT_BG)
    ax.grid(True, alpha=0.3, color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_TEXT)
    ax.spines['bottom'].set_color(COLOR_TEXT)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(colors=COLOR_TEXT, labelsize=11)
    
    # ==================== PLOT ELEMENTS ====================
    # Theoretical PDF - elegant curve
    ax.plot(x, pdf_theoretical, color=COLOR_PDF, lw=3.5,
           label='Theoretical', zorder=10, solid_capstyle='round')
    
    # Histogram - clean bars with transparency
    bars = ax.bar(bin_centers, np.zeros(n_bins),
                 width=bar_width, color=COLOR_HIST,
                 edgecolor=COLOR_PLOT_BG, linewidth=1,
                 alpha=0.6, label='Empirical', zorder=5)
    
    ymax = 1.2 * pdf_theoretical.max()
    ax.set_xlim(samples.min() - 0.05 * (samples.max() - samples.min()),
               samples.max() + 0.05 * (samples.max() - samples.min()))
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r'$x$', fontsize=14, color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Density', fontsize=14, color=COLOR_TEXT, fontweight='600')
    
    # Clean legend
    legend = ax.legend(frameon=False, fontsize=12, loc='upper right')
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT)
    
    # ==================== METRICS RIBBON (Top) ====================
    # Create axes for ribbon at top
    ax_ribbon = fig.add_axes([0.08, 0.86, 0.84, 0.10])
    ax_ribbon.set_xlim(0, 1)
    ax_ribbon.set_ylim(0, 1)
    ax_ribbon.axis('off')
    
    # Ribbon background
    ribbon_bg = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.01",
                              transform=ax_ribbon.transAxes,
                              facecolor=COLOR_RIBBON_BG, edgecolor='none',
                              zorder=0, alpha=0.9)
    ax_ribbon.add_patch(ribbon_bg)
    
    # Title in ribbon
    text_title = ax_ribbon.text(0.02, 0.7, dist_name, fontsize=16,
                               color=COLOR_TEXT, fontweight='bold',
                               va='center')
    
    # Metrics in ribbon (will be updated)
    text_samples = ax_ribbon.text(0.02, 0.25, '', fontsize=11,
                                 color=COLOR_TEXT_LIGHT, va='center',
                                 family='sans-serif')
    
    # Right side metrics
    text_mean = ax_ribbon.text(0.35, 0.7, '', fontsize=10.5,
                              color=COLOR_TEXT, va='center', family='monospace')
    text_std = ax_ribbon.text(0.35, 0.25, '', fontsize=10.5,
                             color=COLOR_TEXT, va='center', family='monospace')
    
    text_ks = ax_ribbon.text(0.65, 0.7, '', fontsize=10.5,
                            color=COLOR_TEXT, va='center', family='monospace')
    text_status = ax_ribbon.text(0.65, 0.25, '', fontsize=11,
                                color=COLOR_SUCCESS, va='center',
                                fontweight='bold')
    
    # Progress bar in ribbon (minimal design)
    progress_y = 0.05
    progress_height = 0.08
    progress_bg = Rectangle((0.02, progress_y), 0.96, progress_height,
                           transform=ax_ribbon.transAxes,
                           facecolor='white', edgecolor=COLOR_GRID,
                           linewidth=1, zorder=1, alpha=0.5)
    ax_ribbon.add_patch(progress_bg)
    
    progress_fill = Rectangle((0.02, progress_y), 0.0, progress_height,
                             transform=ax_ribbon.transAxes,
                             facecolor=COLOR_ACCENT, edgecolor='none',
                             zorder=2, alpha=0.7)
    ax_ribbon.add_patch(progress_fill)
    
    # ==================== ANIMATION UPDATE ====================
    def update(frame_idx):
        n_current = sample_indices[frame_idx]
        current_samples = samples[:n_current]
        
        # Update histogram
        hist_vals, _ = np.histogram(current_samples, bins=bins, density=True)
        
        for bar, height in zip(bars, hist_vals):
            bar.set_height(height)
        
        # Compute statistics
        mean_emp = np.mean(current_samples)
        std_emp = np.std(current_samples, ddof=1)
        ks_stat, ks_pval = scistats.kstest(current_samples, rv.cdf)
        
        # Status
        if ks_stat < 0.02:
            status = "✓ Excellent fit"
            status_color = COLOR_SUCCESS
        elif ks_stat < 0.05:
            status = "✓ Good fit"
            status_color = COLOR_SUCCESS
        elif ks_stat < 0.1:
            status = "~ Moderate fit"
            status_color = COLOR_ACCENT
        else:
            status = "⋯ Converging"
            status_color = COLOR_TEXT_LIGHT
        
        # Update ribbon text
        text_samples.set_text(f'Samples: {n_current:,} / {n_total:,}')
        
        mean_error = abs(mean_emp - mean_theoretical)
        text_mean.set_text(f'μ: {mean_emp:.4f}  (Δ={mean_error:.4f})')
        
        std_error = abs(std_emp - std_theoretical)
        text_std.set_text(f'σ: {std_emp:.4f}  (Δ={std_error:.4f})')
        
        text_ks.set_text(f'KS: {ks_stat:.4f}')
        text_status.set_text(status)
        text_status.set_color(status_color)
        
        # Update progress bar
        progress = n_current / n_total
        progress_fill.set_width(0.96 * progress)
        
        return bars, text_samples, text_mean, text_std, text_ks, text_status
    
    # ==================== BUILD ANIMATION ====================
    anim = FuncAnimation(
        fig, update, frames=frames, interval=1000/fps,
        blit=False, repeat=False
    )
    
    # ==================== SAVE ====================
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    
    if kind.lower() == "mp4":
        writer = FFMpegWriter(fps=fps, bitrate=3000,
                             codec='libx264',
                             extra_args=['-pix_fmt', 'yuv420p'])
        anim.save(out_path, writer=writer, dpi=150)
    else:
        raise ValueError("This function is optimized for MP4 output. Use kind='mp4'")
    
    plt.close(fig)
    print(f"✓ Elegant convergence animation saved to: {out_path}")
    return out_path