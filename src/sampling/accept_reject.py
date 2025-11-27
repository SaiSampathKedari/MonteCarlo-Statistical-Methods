from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.colors import to_rgba

import os
import numpy as np
import scipy.stats as scistats

@dataclass
class AcceptRejectSamples:
    # X-coordinates (proposed Y)
    x_accepted: np.ndarray
    x_rejected: np.ndarray

    # Vertical coords for scatter in (x,u): u = U * M * g(Y)
    u_accepted: np.ndarray
    u_rejected: np.ndarray

    # Helpful for overlays/verification
    f_accepted: np.ndarray     # f(Y) at accepted points
    f_rejected: np.ndarray     # f(Y) at rejected points
    Mg_accepted: np.ndarray    # M*g(Y) at accepted points
    Mg_rejected: np.ndarray    # M*g(Y) at rejected points

    M: float                   # envelope used
    
    # Convenience: acceptance ratio at accepted points
    def r_accepted(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.Mg_accepted > 0, self.f_accepted / self.Mg_accepted, 0.0)

@dataclass
class AcceptRejectEventLog:
    """
    Chronological log of every proposal for animation.
    All arrays are 1-D and aligned index-wise.
    """
    x_all: np.ndarray         # proposed Y
    u_all: np.ndarray         # U * M * g(Y)
    f_all: np.ndarray         # f(Y)
    Mg_all: np.ndarray        # M * g(Y)
    accepted_mask: np.ndarray # boolean array: True if accepted

    # Optional: proposal batch sizes if you want to animate by batches
    batch_sizes: Optional[np.ndarray] = None

    def num_proposals(self) -> int:
        return int(self.x_all.shape[0])

def accept_reject_sampling( 
    num_samples: int,
    proposal_sample_generator: Callable[[int], np.ndarray],          # draws Y ~ g
    proposal_pdf_evaluator: Callable[[np.ndarray], np.ndarray],      # g(Y)
    target_pdf_evaluator: Callable[[np.ndarray], np.ndarray],        # f(Y)
    M: float,
    *,
    return_diagnostics: bool = False,
    return_event_log: bool = False,
    max_rounds: int = 10000
) -> Union[
    AcceptRejectSamples,
    Tuple[AcceptRejectSamples, Dict[str, float]],
    Tuple[AcceptRejectSamples, AcceptRejectEventLog],
    Tuple[AcceptRejectSamples, Dict[str, float], AcceptRejectEventLog]]:
    
    
    assert num_samples > 0, "n must be positive"
    assert M > 0, "M must be positive"
    
    xs_acc, xs_rej = [], []
    us_acc, us_rej = [], []
    f_acc, f_rej = [], []
    Mg_acc, Mg_rej = [], []
    
    # event-log buffers (only filled if requested)
    if return_event_log:
        x_all_buf, u_all_buf, f_all_buf, Mg_all_buf, acc_mask_buf = [], [], [], [], []
        batch_sizes = []
    
    total_props = 0
    remaining = num_samples
    rounds = 0
    
    while remaining > 0:
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError("Exceeded max_rounds; check M and proposal support.")
        
        k = max(1, int(np.ceil(M * remaining)))
        Y = proposal_sample_generator(k)              # proposals
        U = np.random.rand(k)                         # uniforms in [0,1)
        gY = proposal_pdf_evaluator(Y)                # g(Y) >= 0
        fY = target_pdf_evaluator(Y)                  # f(Y) >= 0
        MgY = M * gY
        
        # Support hole check: forbidden if g(Y)=0 but f(Y)>0
        bad_support = (MgY == 0.0) & (fY > 0.0)
        if np.any(bad_support):
            raise RuntimeError("Envelope/support violation: g(y)=0 where f(y)>0. Fix g or increase M.")
        
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(MgY > 0.0, fY / MgY, 0.0)
        r = np.maximum(r, 0.0)
        if r.max(initial=0.0) > 1.0:
            raise RuntimeError("Envelope violated: f(x) > M g(x) somewhere. Increase M or choose a better g.")
        
        accept_mask = U < np.minimum(r, 1.0)
        reject_mask = ~accept_mask
        
        # vertical coords for plotting: u = U * M g(Y)
        u_plot = U * MgY
        
        # log events if requested
        if return_event_log:
            x_all_buf.append(Y)
            u_all_buf.append(u_plot)
            f_all_buf.append(fY)
            Mg_all_buf.append(MgY)
            acc_mask_buf.append(accept_mask)
            batch_sizes.append(k)
        
        # split and append
        Xa, Xr = Y[accept_mask], Y[reject_mask]
        ua, ur = u_plot[accept_mask], u_plot[reject_mask]
        fa, fr = fY[accept_mask], fY[reject_mask]
        Mga, Mgr = MgY[accept_mask], MgY[reject_mask]
        
        if Xa.size:
            xs_acc.append(Xa);  us_acc.append(ua);  f_acc.append(fa);  Mg_acc.append(Mga)
            remaining -= Xa.shape[0]
        if Xr.size:
            xs_rej.append(Xr);  us_rej.append(ur);  f_rej.append(fr);  Mg_rej.append(Mgr)
        
        total_props += k
    
    # concatenate and truncate accepted to exactly n
    x_acc = np.concatenate(xs_acc, axis=0)[:num_samples]
    u_acc = np.concatenate(us_acc, axis=0)[:num_samples]
    f_a  = np.concatenate(f_acc, axis=0)[:num_samples]
    Mg_a = np.concatenate(Mg_acc, axis=0)[:num_samples]

    x_rej = np.concatenate(xs_rej, axis=0) if xs_rej else np.empty((0,), dtype=float)
    u_rej = np.concatenate(us_rej, axis=0) if us_rej else np.empty((0,), dtype=float)
    f_r   = np.concatenate(f_rej, axis=0) if f_rej else np.empty((0,), dtype=float)
    Mg_r  = np.concatenate(Mg_rej, axis=0) if Mg_rej else np.empty((0,), dtype=float)
    
    result = AcceptRejectSamples(
        x_accepted=x_acc,
        x_rejected=x_rej,
        u_accepted=u_acc,
        u_rejected=u_rej,
        f_accepted=f_a,
        f_rejected=f_r,
        Mg_accepted=Mg_a,
        Mg_rejected=Mg_r,
        M=M
    )
    
    # build diagnostics if requested
    diag = None
    if return_diagnostics:
        diag = {
            "num_samples_requested": int(num_samples),
            "total_proposals": int(total_props),
            "acceptance_rate": float(num_samples) / float(total_props),
            "expected_acceptance": 1.0 / float(M),
            "rounds": float(rounds),
        }
    
    # build event log if requested
    evlog = None
    if return_event_log:
        x_all  = np.concatenate(x_all_buf) if x_all_buf else np.empty((0,), dtype=float)
        u_all  = np.concatenate(u_all_buf) if u_all_buf else np.empty((0,), dtype=float)
        f_all  = np.concatenate(f_all_buf) if f_all_buf else np.empty((0,), dtype=float)
        Mg_all = np.concatenate(Mg_all_buf) if Mg_all_buf else np.empty((0,), dtype=float)
        accepted_mask = np.concatenate(acc_mask_buf).astype(bool) if acc_mask_buf else np.zeros((0,), dtype=bool)
        evlog = AcceptRejectEventLog(
            x_all=x_all, u_all=u_all, f_all=f_all, Mg_all=Mg_all,
            accepted_mask=accepted_mask,
            batch_sizes=np.asarray(batch_sizes, dtype=int) if batch_sizes else None
        )
    
    # return tuples based on flags
    if return_diagnostics and return_event_log:
        return result, diag, evlog
    if return_diagnostics:
        return result, diag
    if return_event_log:
        return result, evlog
    return result

def accept_reject_scatter_plot(
    samples,                      # AcceptRejectSamples
    target_rv,                    # scipy.stats.rv_continuous (for f)
    proposal_rv,                  # scipy.stats.rv_continuous (for g)
    N: int = 600,
    ppm_thresh: float = 1e-4,
    save_path: str | None = None,
) -> None:
    """
    Scatter visualization for Accept–Reject in the (x,u)-plane.
      • Green dots: accepted proposals at (x, u = U * M * g(x))
      • Red dots  : rejected proposals at (x, u = U * M * g(x))
      • Curves: f(x) (target) and M g(x) (envelope)
      • Labels: domain 𝓧 and region 𝓛 = {(y,u): 0 < u < M g(y)}
    """
    # ---- Sanity checks ----
    M = float(samples.M)
    assert M > 0, "M must be positive."

    # Accepted
    xa = np.asarray(samples.x_accepted).reshape(-1)
    ua = np.asarray(samples.u_accepted).reshape(-1)
    fa = np.asarray(samples.f_accepted).reshape(-1)
    Mga = np.asarray(samples.Mg_accepted).reshape(-1)

    # Rejected
    xr = np.asarray(samples.x_rejected).reshape(-1)
    ur = np.asarray(samples.u_rejected).reshape(-1)
    fr = np.asarray(samples.f_rejected).reshape(-1)
    Mgr = np.asarray(samples.Mg_rejected).reshape(-1)

    # Optional correctness checks at sample points
    if xa.size:
        if not (np.all(np.isfinite(ua)) and np.all(np.isfinite(fa)) and np.all(np.isfinite(Mga))):
            raise ValueError("Non-finite values in accepted batches.")
        if np.any(ua > fa + 1e-14):
            raise RuntimeError("Accepted point above f(x). Check implementation.")
    if xr.size:
        if not (np.all(np.isfinite(ur)) and np.all(np.isfinite(fr)) and np.all(np.isfinite(Mgr))):
            raise ValueError("Non-finite values in rejected batches.")
        if np.any(ur < fr - 1e-14):
            # Not strictly required: rejected can lie below f if U>r but visualize warning
            pass

    # ---- Build plotting grid over union support (curves + samples) ----
    def safe_ppf(rv, q, fallback=1e-8):
        x = rv.ppf(q)
        if not np.isfinite(x):
            x = rv.ppf(np.clip(q, fallback, 1.0 - fallback))
        return x

    q_lo, q_hi = ppm_thresh, 1.0 - ppm_thresh
    t_lo, t_hi = safe_ppf(target_rv, q_lo), safe_ppf(target_rv, q_hi)
    g_lo, g_hi = safe_ppf(proposal_rv, q_lo), safe_ppf(proposal_rv, q_hi)

    data_min = np.min([t_lo, g_lo, xa.min() if xa.size else np.inf, xr.min() if xr.size else np.inf])
    data_max = np.max([t_hi, g_hi, xa.max() if xa.size else -np.inf, xr.max() if xr.size else -np.inf])

    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_min >= data_max:
        data_min, data_max = -10.0, 10.0

    x = np.linspace(data_min, data_max, N)
    f_x = target_rv.pdf(x)
    Mg_x = M * proposal_rv.pdf(x)

    if np.any(~np.isfinite(f_x)) or np.any(~np.isfinite(Mg_x)):
        raise ValueError("Non-finite pdf values on grid.")
    if np.any(Mg_x < f_x):
        frac = np.mean(Mg_x < f_x)
        raise ValueError(f"Envelope violation on grid: M g(x) < f(x) on {frac*100:.2f}% of points.")

    # ---- Plot ----
    plt.rcParams.update({
        "axes.edgecolor": "black",
        "axes.linewidth": 1.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13
    })
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_facecolor("white")
    ax.grid(False)

    # Curves
    ax.plot(x, f_x,  color="#1F4B99", lw=3.0, label=r"$f(x)$ (target)")
    ax.plot(x, Mg_x, color="#A85E00", lw=3.0, label=rf"$M\,g(x)$ (envelope)")

    # Scatter: accepted (green), rejected (red)
    if xr.size:
        ax.scatter(xr, ur, s=14, alpha=0.85, color="#D32F2F", edgecolors="none", label="Rejected")
    if xa.size:
        ax.scatter(xa, ua, s=16, alpha=0.95, color="#2E7D32", edgecolors="none", label="Accepted")

    # Labels, legend, axes
    ax.set_xlabel(r"$\mathcal{X}$", fontsize=16)
    ax.set_ylabel(r"Vertical axis $u$", fontsize=16)
    ax.set_title(r"Accept-Reject: Samples in $(x,u)$ with $f(x)$ and $M g(x)$",
                 fontsize=20, pad=14, weight="bold")

    # Region label 𝓛
    # xi = x[len(x)//4]
    # ui = 0.85 * np.max(Mg_x)
    # ax.text(xi, ui, r"$\mathcal{L}=\{(y,u):\,0<u<Mg(y)\}$",
    #         fontsize=14, color="black",
    #         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))

    ax.legend(loc="upper right", fontsize=13, frameon=False)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.1 * np.max(Mg_x))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


def accept_reject_plot(
    target_rv: scistats.rv_continuous,
    proposal_rv: scistats.rv_continuous,
    M: float,
    N: int = 600,
    ppm_thresh: float = 1e-4,
    save_path: str = None
) -> None:
    """
    High-contrast Accept–Reject geometry plot:
      - f(x) (target) shown in purple with solid fill under it (accepted region)
      - M g(x) (envelope) shown in gold with solid fill between M g(x) and f(x) (rejected region)
      - union support from both target and proposal (no cutoffs)
      - labels for L and \mathcal{X}

    Parameters
    ----------
    target_rv : scipy.stats.rv_continuous
    proposal_rv : scipy.stats.rv_continuous
    M : float
        Envelope constant, must satisfy f(x) <= M g(x) on the plotted domain.
    N : int
        Number of x grid points.
    ppm_thresh : float
        Tail trimming for ppf to avoid infinities.
    save_path : str or None
        If provided, saves the figure to this path.
    """

    assert M > 0, "M must be positive."

    def safe_ppf(rv, q, fallback=1e-8):
        """Robust ppf that retries with a safer bound if needed."""
        x = rv.ppf(q)
        if not np.isfinite(x):
            x = rv.ppf(np.clip(q, fallback, 1.0 - fallback))
        return x

    # Build a union support from BOTH distributions to avoid cutting off curves
    q_lo, q_hi = ppm_thresh, 1.0 - ppm_thresh
    t_lo, t_hi = safe_ppf(target_rv, q_lo), safe_ppf(target_rv, q_hi)
    g_lo, g_hi = safe_ppf(proposal_rv, q_lo), safe_ppf(proposal_rv, q_hi)

    x_min = np.nanmin([t_lo, g_lo])
    x_max = np.nanmax([t_hi, g_hi])

    # Handle degenerate/infinite endpoints gracefully
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
        # fallback to a symmetric numeric window if necessary
        x_min, x_max = -10.0, 10.0

    x = np.linspace(x_min, x_max, N)

    # Evaluate PDFs
    f_x = target_rv.pdf(x)
    g_x = proposal_rv.pdf(x)
    Mg_x = M * g_x

    # Keep only finite values
    mask = np.isfinite(f_x) & np.isfinite(Mg_x)
    x, f_x, Mg_x = x[mask], f_x[mask], Mg_x[mask]

    if x.size == 0:
        raise ValueError("No finite plotting region found for the given distributions.")

    # Envelope validity check on the plotted domain
    if np.any(Mg_x < f_x):
        bad_frac = np.mean(Mg_x < f_x)
        raise ValueError(
            f"M * g(x) must dominate f(x) on the plotting grid. "
            f"Violation on {bad_frac*100:.2f}% of points. Increase M or choose a better proposal."
        )

    # --- Styling: poster-clean, no grid, bold lines/fills ---
    plt.rcParams.update({
        "axes.edgecolor": "black",
        "axes.linewidth": 1.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13
    })

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_facecolor("white")
    ax.grid(False)

    # Solid base fill for envelope region \mathcal{L}
    ax.fill_between(x, 0, Mg_x, color="#FFD500", alpha=1.0)  # envelope area (yellow)

    # Solid fill for accepted region under f(x)
    ax.fill_between(x, 0, f_x, color="#7B1FA2", alpha=1.0)  # accepted area (purple)

    # Distinct, colored curves with labels
    ax.plot(x, f_x, color="#4A148C", lw=3.0, label=r"$f(x)$ (target)")
    ax.plot(x, Mg_x, color="#FF8F00", lw=3.0, label=rf"$M\,g(x)$ (envelope)")

    # Axis labels (math-consistent with your notes)
    ax.set_xlabel(r"$\mathcal{X}$", fontsize=16)   # domain label
    ax.set_ylabel(r"Density", fontsize=16)

    # Title
    ax.set_title(r"Accept--Reject Sampling: Geometric View", fontsize=20, pad=14, weight="bold")

    # Region label \mathcal{L}
    # Place the text where the envelope is clearly above f(x)
    xi = x[len(x)//4]
    ui = 0.8 * np.max(Mg_x)  # near the top for visibility
    ax.text(xi, ui, r"$\mathcal{L}=\{(y,u):\,0<u<Mg(y)\}$",
            fontsize=14, color="black", bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    # Legend
    leg = ax.legend(loc="upper right", fontsize=13, frameon=False)
    for line in leg.get_lines():
        line.set_linewidth(3.0)

    # Bounds
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.1 * np.max(Mg_x))

    # Optional save
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

def animate_accept_reject_scatter(
    evlog,                      # AcceptRejectEventLog instance
    target_rv,                  # scipy.stats.rv_continuous
    proposal_rv,                # scipy.stats.rv_continuous
    M: float,
    dist_name: str = "Accept-Reject Sampling",
    out_path: str = "animations/accept_reject_scatter.gif",
    N: int = 600,
    ppm_thresh: float = 1e-3,
    frames: int = 300,
    fps: int = 25,
    interval_ms: int = 40,
    figsize=(9, 5.5),
    facecolor="white",
    alpha_points=0.85,
    kind: str = "gif"
) -> str:
    """
    Animate Accept-Reject sampling:
      - Shows both target f(x) and envelope M*g(x)
      - Sequentially reveals sampled points
      - Accepted = green, Rejected = red

    Parameters
    ----------
    evlog : AcceptRejectEventLog
        Full event log with x_all, u_all, f_all, Mg_all, accepted_mask
    target_rv, proposal_rv : scipy.stats.rv_continuous
        Used for plotting theoretical f(x) and M*g(x)
    M : float
        Envelope constant.
    dist_name : str
        Title label for the animation.
    out_path : str
        File path to save output (GIF or MP4).
    """

    assert isinstance(M, (int, float)) and M > 0, "M must be positive."

    # --- Compute plotting range from both PDFs ---
    x_min = np.nanmin([target_rv.ppf(ppm_thresh), proposal_rv.ppf(ppm_thresh)])
    x_max = np.nanmax([target_rv.ppf(1 - ppm_thresh), proposal_rv.ppf(1 - ppm_thresh)])
    x = np.linspace(x_min, x_max, N)
    f_x = target_rv.pdf(x)
    Mg_x = M * proposal_rv.pdf(x)

    ymax = 1.15 * np.nanmax(Mg_x)

    # --- Extract event log data ---
    x_all, u_all = evlog.x_all, evlog.u_all
    accepted_mask = evlog.accepted_mask
    n_total = len(x_all)
    if n_total == 0:
        raise ValueError("Empty event log — no samples to animate.")

    # --- Map indices to frames ---
    step = max(1, n_total // frames)
    frame_points = np.arange(0, n_total, step)
    frames = len(frame_points)

    # --- Figure setup ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(facecolor)

    # Curves
    ax.plot(x, f_x, color="#4A148C", lw=2.5, label=r"$f(x)$ (target)")
    ax.plot(x, Mg_x, color="#FF8F00", lw=2.5, label=r"$M g(x)$ (envelope)")

    # Scatter placeholders
    accepted_scatter = ax.scatter([], [], color="#1B5E20", s=25, alpha=alpha_points, label="Accepted")
    rejected_scatter = ax.scatter([], [], color="#B71C1C", s=25, alpha=alpha_points, label="Rejected")

    # Labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$u$", fontsize=14)
    ax.set_title(dist_name, fontsize=16, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=11, loc="upper right")

    # --- Animation update function ---
    def update(frame):
        upto = frame_points[frame]
        acc_idx = np.where(accepted_mask[:upto])[0]
        rej_idx = np.where(~accepted_mask[:upto])[0]

        accepted_scatter.set_offsets(np.column_stack((x_all[acc_idx], u_all[acc_idx])))
        rejected_scatter.set_offsets(np.column_stack((x_all[rej_idx], u_all[rej_idx])))

        ax.set_title(f"{dist_name}\nSamples Shown: {upto}/{n_total}", fontsize=15, pad=12)
        return accepted_scatter, rejected_scatter

    # --- Build animation ---
    anim = FuncAnimation(
        fig, update, frames=frames, interval=interval_ms, blit=False, repeat=False
    )

    # --- Save ---
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