"""Shared figures from compact configuration and aggregate MC score tables."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .scores import load_scores
from .notation import context_math



def compare(run_dir, output, baselines, ours, axis, *, labels=None,
            non_sof=(), formats=("png", "pdf"),
            policy_colors=None, overall_log_left=None):
    """Plot published configuration and aggregate scores directly."""
    display = dict(labels or {})
    methods = baselines + ours
    # TLE (time limit exceeded) runs have no measurements; scores use successful runs only.
    # Missing runs are not filled with zeros; score-table counts record the available coverage.
    context_scores, overall_scores = load_scores(run_dir, ("L", axis))
    output.mkdir(parents=True, exist_ok=True)
    lookup = {r['policy']: r for r in overall_scores}
    summary = [lookup[p] for p in methods]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    policy_colors = policy_colors or {}
    colors = [policy_colors.get(p, "#D55E00" if p in ours else "#0072B2") for p in methods]
    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, max(5, .36 * len(methods) + 1.6)))
    values = np.array([r['fitness'] for r in summary])
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Log aggregate bars require positive finite fitness")
    lower = 10. ** np.floor(np.log10(float(values.min()) * .85))
    if overall_log_left is not None:
        lower = float(overall_log_left)
        if not np.isfinite(lower) or not 0 < lower < float(values.min()):
            raise ValueError("Log left limit must be positive and below every fitness")
    ax.barh(y, values - lower, left=lower, color=colors, height=.32)
    for i, value in enumerate(values):
        ax.text(1.02, i, f'{value:.3f}',
                transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    groups = {p: ("Our" if p in ours else "non-SOF" if p in non_sof else "SOF")
              for p in methods}
    names = [f'{display.get(p, p)} ({groups[p]})' for p in methods]
    ax.set_yticks(y, names)
    for tick in ax.get_yticklabels():
        tick.set_color("black")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("aggregate fitness")
    if overall_log_left is not None:
        from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullLocator
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.,)))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(lower, float(values.max()) * 1.25)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=.30, right=.82, bottom=.15, top=.97)
    save(fig, output / "overall", formats=formats)

    contexts = sorted({(r['L'], r[axis]) for r in context_scores if r['policy'] in methods})
    lookup = {(r['policy'], r['L'], r[axis]): r['fitness'] for r in context_scores}
    matrix = np.array([[lookup[p, l, k] for l, k in contexts] for p in methods])
    from matplotlib.colors import SymLogNorm, LinearSegmentedColormap
    # Nonnegative best-observed gaps; logarithmic colors reveal small differences.
    limit = max(1., float(np.max(np.abs(matrix))))
    cmap = LinearSegmentedColormap.from_list('fitness', ['#fcfcfa', '#f3d1c0', '#a83e32'])
    norm = SymLogNorm(linthresh=.1, linscale=.6, vmin=0, vmax=limit, base=10)
    fig, ax = plt.subplots(figsize=(12.5, max(5.7, .43 * len(methods) + 2.2)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    for i in range(len(methods)):
        for j in range(len(contexts)):
            v = matrix[i, j]
            label = "0.00" if abs(v) < .005 else f"{v:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    color="white" if float(norm(v)) > .7 else "#263442")
    ax.set_yticks(y, names)
    for tick, policy in zip(ax.get_yticklabels(), methods):
        tick.set_color("black")
        if policy in ours:
            tick.set_fontweight("bold")
    ax.set_xticks(range(len(contexts)), [context_math(axis, k) for _, k in contexts])
    ax.tick_params(axis="both", which="both", length=0, pad=9)
    ax.set_xticks(np.arange(-.5, len(contexts), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    for l in sorted({l for l, _ in contexts}):
        cols = [j for j, (ll, _) in enumerate(contexts) if ll == l]
        midpoint = (min(cols) + max(cols)) / 2
        ax.text(midpoint, 1.025, context_math('L', l), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="#263442")
        if min(cols):
            ax.axvline(min(cols) - .5, color="white", lw=5)
    ax.axhline(len(baselines) - .5, color="white", lw=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=.027, pad=.025)
    ticks = [v for v in (0, .1, 1, 10, 100, 1000) if v <= limit]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{v:g}%" for v in ticks])
    cbar.set_label("Robust fitness (%) · symlog color scale", labelpad=10)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8)
    fig.subplots_adjust(left=.30, bottom=.12, top=.97, right=.91)
    save(fig, output / "contexts", formats=formats)
    return summary


def save(fig, stem, *, formats=("png", "pdf")):
    for ext in formats:
        fig.savefig(stem.with_suffix("." + ext), dpi=180, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)
