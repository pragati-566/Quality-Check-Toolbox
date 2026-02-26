"""
visualize.py — Plotting utilities for the QC Toolbox QEI demo.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# --------------------------------------------------------------------------- #
# Colour maps                                                                  #
# --------------------------------------------------------------------------- #
CBF_CMAP   = "hot"
SCORE_CMAP = {"excellent": "#2ecc71", "average": "#f39c12", "poor": "#e74c3c"}


def _mid_slice(volume: np.ndarray) -> np.ndarray:
    """Return the axial mid-slice of a 3-D volume."""
    return volume[:, :, volume.shape[2] // 2]


def plot_cbf_slice(cbf_map: np.ndarray, title: str, ax: plt.Axes,
                   vmin: float = -10, vmax: float = 100) -> None:
    """
    Draw the mid-axial slice of a CBF map on *ax*.

    Parameters
    ----------
    cbf_map : 3-D ndarray
    title   : subplot title
    ax      : matplotlib Axes
    vmin, vmax : colour-scale limits (ml/100g/min)
    """
    im = ax.imshow(_mid_slice(cbf_map).T, origin="lower",
                   cmap=CBF_CMAP, vmin=vmin, vmax=vmax,
                   interpolation="bilinear")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ml/100g/min")


def plot_qei_report(results: list[dict], save_path: str = "qei_report.png") -> None:
    """
    Create and save a combined figure:
        • Top row    — mid-axial CBF slices for each case
        • Bottom row — bar chart of QEI scores

    Parameters
    ----------
    results   : list of dicts, each with keys:
                    label, cbf_map, qei, pss, di, n_gm
    save_path : file path for the output PNG
    """
    n   = len(results)
    fig = plt.figure(figsize=(5 * n, 9), facecolor="#0f1117")
    gs  = gridspec.GridSpec(2, n, figure=fig,
                             hspace=0.35, wspace=0.30,
                             height_ratios=[1.3, 1])

    # ── Top row: CBF slices ──────────────────────────────────────────────── #
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#0f1117")
        plot_cbf_slice(r["cbf_map"],
                       f"{r['label']}\nQEI = {r['qei']:.3f}", ax)

    # ── Bottom row: score bar chart ──────────────────────────────────────── #
    ax_bar = fig.add_subplot(gs[1, :])
    ax_bar.set_facecolor("#1a1d27")

    labels = [r["label"] for r in results]
    qeis   = [r["qei"]   for r in results]
    colors = [SCORE_CMAP.get(r["label"].lower(), "#3498db") for r in results]

    bars = ax_bar.bar(labels, qeis, color=colors, width=0.4,
                      edgecolor="white", linewidth=0.5, zorder=3)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("QEI Score", color="white", fontsize=12)
    ax_bar.set_title("Quality Evaluation Index — Summary", color="white",
                     fontsize=13, fontweight="bold")
    ax_bar.tick_params(colors="white")
    ax_bar.spines[:].set_color("#444")
    ax_bar.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.6, zorder=0)
    ax_bar.set_axisbelow(True)

    # Add value labels on bars
    for bar, qei in zip(bars, qeis):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{qei:.3f}", ha="center", va="bottom",
                    color="white", fontsize=12, fontweight="bold")

    # Component breakdown as annotation
    for i, r in enumerate(results):
        note = (f"pss={r['pss']:.3f}  "
                f"DI={r['di']:.2f}  "
                f"neg={r['n_gm']:.1%}")
        ax_bar.text(i, -0.10, note, ha="center", va="top",
                    transform=ax_bar.get_xaxis_transform(),
                    fontsize=8, color="#aaa")

    fig.text(0.5, 0.01,
             "QEI formula: Dolui et al. 2024, JMRI — doi:10.1002/jmri.29308",
             ha="center", fontsize=8, color="#666")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Report saved → {save_path}")


def print_report(results: list[dict]) -> None:
    """
    Print a formatted table of QEI results to stdout.

    Parameters
    ----------
    results : list of dicts (same format as plot_qei_report)
    """
    # Header
    line = "─" * 60
    print(f"\n{'Quality Check Toolbox v1.0 — QEI Results':^60}")
    print(f"{'(Dolui et al. 2024, JMRI)':^60}\n")
    print(line)
    print(f"{'Case':<12} {'QEI':>6}  {'pss':>6}  {'DI':>6}  {'neg GM':>8}  {'Grade'}")
    print(line)

    for r in results:
        qei = r["qei"]
        grade = (
            "Excellent" if qei >= 0.75
            else "Average"   if qei >= 0.50
            else "Poor"
        )
        print(
            f"{r['label']:<12} "
            f"{r['qei']:>6.4f}  "
            f"{r['pss']:>6.4f}  "
            f"{r['di']:>6.2f}  "
            f"{r['n_gm']:>7.1%}  "
            f"{grade}"
        )
    print(line)
    print("\nComponents:\n"
          "  pss    — structural similarity (Pearson r with pseudo-CBF)\n"
          "  DI     — index of dispersion (spatial variability)\n"
          "  neg GM — fraction of GM voxels with negative CBF\n"
          "  QEI  ∈ [0, 1] — higher is better\n")
