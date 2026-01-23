import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
import os
from typing import Callable


def animate_mcmc_simple(
    samples: np.ndarray,
    accept_mask: np.ndarray,
    eval_on_grid_func: Callable,
    target_func: Callable,
    algorithm_name: str = "DRAM",
    title: str = "DRAM: Delayed Rejection Adaptive Metropolis for Banana Distribution",
    out_path: str = "animations/mcmc.mp4",
    xlim=(-3, 3),
    ylim=(-10, 2),
    frames: int = 400,
    fps: int = 30,
    trail_length: int = 15,
    figsize=(11, 9),
) -> str:
    """
    Professional MCMC animation for website presentation.
    
    Parameters
    ----------
    samples : np.ndarray
        All MCMC samples (N, 2)
    accept_mask : np.ndarray
        Boolean mask (N,) - True where accepted
    eval_on_grid_func : Callable
        Your eval_on_2D_grid function
    target_func : Callable  
        Your target PDF function
    algorithm_name : str
        Short name for stats panel (e.g., "DRAM")
    title : str
        Full title displayed above the plot
    out_path : str
        Output file path
    xlim, ylim : tuple
        Plot limits
    frames : int
        Number of animation frames
    fps : int
        Frames per second
    trail_length : int
        Number of recent points in trail
    figsize : tuple
        Figure size (width, height)
    
    Returns
    -------
    str
        Path to saved animation file
    """
    
    n_total = len(samples)
    
    # ==================== EVALUATE BANANA ON GRID ====================
    print("Evaluating banana distribution on grid...")
    xgrid = np.linspace(xlim[0], xlim[1], 400)
    ygrid = np.linspace(ylim[0], ylim[1], 400)
    
    XX, YY, pdf_evals = eval_on_grid_func(xgrid, ygrid, target_func)
    
    # ==================== FIGURE SETUP ====================
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Add title at the top
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98, color='#2C3E50')
    
    # Tight layout - no wasted space
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.06,
                       hspace=0.08, wspace=0.08)
    
    # 2x2 grid with proper proportions
    gs = GridSpec(2, 2, figure=fig,
                 width_ratios=[4, 1.2],      # Extra space for θ₂ marginal
                 height_ratios=[0.8, 4],     # More space for main plot
                 hspace=0.08, wspace=0.08)
    
    ax_marg_x = fig.add_subplot(gs[0, 0])    # Top-left: marginal x₁
    ax_stats = fig.add_subplot(gs[0, 1])     # Top-right: stats
    ax_joint = fig.add_subplot(gs[1, 0])     # Bottom-left: joint
    ax_marg_y = fig.add_subplot(gs[1, 1])    # Bottom-right: marginal x₂
    
    # ==================== JOINT PLOT ====================
    # Beautiful magma colormap with more levels
    ax_joint.contourf(XX, YY, pdf_evals, levels=50, cmap='magma')
    ax_joint.contour(XX, YY, pdf_evals, levels=20, colors='cyan',
                    linewidths=0.3, alpha=0.4)
    
    # ALL visited samples (accumulating light dots)
    scatter_all = ax_joint.scatter([], [], s=4, c='cyan', alpha=0.12, zorder=5,
                                  edgecolors='none')
    
    # Recent trail (thicker, bright red, fading)
    trail_lines = []
    for i in range(trail_length):
        alpha = (i + 1) / trail_length
        line, = ax_joint.plot([], [], '-', color='#FF3366', lw=3.5,  # Thicker!
                             alpha=alpha * 0.9, zorder=10-i,
                             solid_capstyle='round')
        trail_lines.append(line)
    
    # Current point (bright yellow, impossible to miss)
    current_point, = ax_joint.plot([], [], 'o', color='#FFFF00', markersize=14,
                                   markeredgecolor='white', markeredgewidth=3,
                                   zorder=20)
    
    ax_joint.set_xlim(xlim)
    ax_joint.set_ylim(ylim)
    ax_joint.set_xlabel(r'$x_1$', fontsize=16, fontweight='bold')  # Changed to x₁
    ax_joint.set_ylabel(r'$x_2$', fontsize=16, fontweight='bold')  # Changed to x₂
    ax_joint.tick_params(labelsize=12)
    ax_joint.grid(True, alpha=0.15, linewidth=0.5, color='white')  # Subtle grid
    
    # ==================== MARGINAL X (TOP) ====================
    ax_marg_x.set_xlim(xlim)
    ax_marg_x.set_ylabel('Density', fontsize=12, fontweight='600')
    ax_marg_x.tick_params(labelbottom=False, labelsize=11)
    ax_marg_x.spines['bottom'].set_visible(False)
    
    # Histogram
    hist_x_bins = np.linspace(xlim[0], xlim[1], 60)
    hist_x_centers = 0.5 * (hist_x_bins[:-1] + hist_x_bins[1:])
    hist_x_width = hist_x_bins[1] - hist_x_bins[0]
    
    bars_x = ax_marg_x.bar(hist_x_centers, np.zeros(len(hist_x_centers)),
                          width=hist_x_width, color='#3498DB', alpha=0.85,
                          edgecolor='#2C3E50', linewidth=0.6)
    
    # Dynamic y-limit will be set during animation
    ax_marg_x.set_ylim(0, 0.7)
    
    # ==================== MARGINAL Y (RIGHT) ====================
    ax_marg_y.set_ylim(ylim)
    ax_marg_y.set_xlabel('Density', fontsize=12, fontweight='600')
    ax_marg_y.tick_params(labelleft=False, labelsize=11)
    ax_marg_y.spines['left'].set_visible(False)
    
    # Histogram
    hist_y_bins = np.linspace(ylim[0], ylim[1], 60)
    hist_y_centers = 0.5 * (hist_y_bins[:-1] + hist_y_bins[1:])
    hist_y_height = hist_y_bins[1] - hist_y_bins[0]
    
    bars_y = ax_marg_y.barh(hist_y_centers, np.zeros(len(hist_y_centers)),
                           height=hist_y_height, color='#3498DB', alpha=0.85,
                           edgecolor='#2C3E50', linewidth=0.6)
    
    # Dynamic x-limit will be set during animation
    ax_marg_y.set_xlim(0, 0.25)
    
    # ==================== STATS PANEL ====================
    ax_stats.axis('off')
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    
    # Algorithm name
    ax_stats.text(0.5, 0.78, algorithm_name, ha='center', va='top',
                 fontsize=17, fontweight='bold', color='#2C3E50')
    
    # Stats text (will be updated)
    text_samples = ax_stats.text(0.5, 0.52, '', ha='center', va='top',
                                fontsize=13, fontweight='600', color='#34495E')
    text_accept = ax_stats.text(0.5, 0.25, '', ha='center', va='top',
                               fontsize=14, color='#27AE60', fontweight='bold')
    
    # ==================== FRAME MAPPING ====================
    frame_indices = np.linspace(trail_length, n_total - 1, frames, dtype=int)
    frames = len(frame_indices)
    
    # ==================== ANIMATION UPDATE ====================
    def update(frame_idx):
        current_idx = frame_indices[frame_idx]
        current_samples = samples[:current_idx + 1]
        
        # Update ALL accumulated samples
        scatter_all.set_offsets(current_samples)
        
        # Update fading trail
        for i, line in enumerate(trail_lines):
            start_idx = max(0, current_idx - trail_length + i)
            end_idx = current_idx - trail_length + i + 1
            
            if end_idx > start_idx and end_idx <= current_idx + 1:
                trail_segment = samples[start_idx:end_idx + 1]
                if len(trail_segment) > 1:
                    line.set_data(trail_segment[:, 0], trail_segment[:, 1])
                else:
                    line.set_data([], [])
            else:
                line.set_data([], [])
        
        # Update current point
        current_point.set_data([samples[current_idx, 0]], [samples[current_idx, 1]])
        
        # Update marginal histograms
        hist_x, _ = np.histogram(current_samples[:, 0], bins=hist_x_bins, density=True)
        hist_y, _ = np.histogram(current_samples[:, 1], bins=hist_y_bins, density=True)
        
        for bar, h in zip(bars_x, hist_x):
            bar.set_height(h)
        
        for bar, h in zip(bars_y, hist_y):
            bar.set_width(h)
        
        # Dynamically adjust y-marginal x-limit to prevent overflow
        max_y_density = np.max(hist_y) if len(hist_y) > 0 else 0.1
        ax_marg_y.set_xlim(0, max_y_density * 1.15)  # 15% padding
        
        # Update stats
        n_accepted = np.sum(accept_mask[:current_idx + 1])
        accept_rate = n_accepted / (current_idx + 1)
        
        text_samples.set_text(f'Samples: {current_idx + 1:,}')
        text_accept.set_text(f'Accept Rate:\n{accept_rate:.1%}')
        
        return ([scatter_all, current_point] + trail_lines + 
                list(bars_x) + list(bars_y) + [text_samples, text_accept])
    
    # ==================== BUILD ANIMATION ====================
    print(f"Creating animation with {frames} frames...")
    anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps,
                        blit=False, repeat=False)
    
    # ==================== SAVE ====================
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    print("Saving animation...")
    writer = FFMpegWriter(fps=fps, bitrate=4000, codec='libx264',
                         extra_args=['-pix_fmt', 'yuv420p'])
    anim.save(out_path, writer=writer, dpi=150)  # Higher DPI for website
    
    plt.close(fig)
    print(f"✓ Animation saved to: {out_path}")
    print(f"  Duration: {frames/fps:.1f} seconds")
    print(f"  Resolution: {int(figsize[0]*150)} x {int(figsize[1]*150)} pixels")
    return out_path