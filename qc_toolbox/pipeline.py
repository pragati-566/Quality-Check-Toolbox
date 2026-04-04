"""
pipeline.py — Main QC pipeline runner for real BIDS ASL data.

Orchestrates:
  1. BIDS subject discovery
  2. Per-subject data loading (bids_loader)
  3. Tissue mask derivation (tissue_masks)
  4. QEI computation (qei)
  5. Results aggregation and flagging
  6. Summary report (CSV + optional plots)

Usage (Python):
    from qc_toolbox.pipeline import run_pipeline
    results = run_pipeline(bids_root="./data/ds004114", output_dir="./qc_output")

Usage (CLI):
    python -m qc_toolbox.pipeline --bids ./data/ds004114 --output ./qc_output
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .bids_loader   import iter_dataset, ASLSubject
from .tissue_masks  import masks_from_cbf
from .qei           import compute_qei


# ──────────────────────────────────────────────────────────────────────────────
# Threshold presets
# ──────────────────────────────────────────────────────────────────────────────
#
# DEFAULT: tuned for mean control−label difference maps (raw ASL proxy), CBF
# percentile masks — DI and QEI ranges differ from quantified CBF (e.g. ASLPrep).
#
# STRICT_QUANTIFIED_CBF: stricter cut-offs when the input map is true CBF
# (ml/100g/min) from a full quantification pipeline + proper segmentation.

# Pass/fail uses QEI only. PSS, DI, n_gm stay in CSV for debugging — they are
# already folded into QEI, so separate cut-offs would double-count.

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "qei_min":           0.30,
    "mean_gm_cbf_min":   float("-inf"),
    "mean_gm_cbf_max":   float("inf"),
}

STRICT_QUANTIFIED_CBF_THRESHOLDS: dict[str, Any] = {
    "qei_min":           0.70,
    "mean_gm_cbf_min":   10.0,
    "mean_gm_cbf_max":   120.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# Per-subject QC
# ──────────────────────────────────────────────────────────────────────────────

def _qc_subject(subject: ASLSubject, thresholds: dict) -> dict:
    """
    Run the full QC pipeline for one subject and return a result dictionary.
    """
    cbf = subject.cbf_map

    # 1. Derive tissue masks from the CBF map
    gm_mask, wm_mask, csf_mask = masks_from_cbf(cbf)

    # 2. Compute QEI (5 mm FWHM smoothing; affine for voxel size)
    qei_result = compute_qei(
        cbf_map  = cbf,
        gm_mask  = gm_mask,
        wm_mask  = wm_mask,
        csf_mask = csf_mask,
        affine   = subject.affine,
    )

    # 3. Additional QC metrics
    mean_gm_cbf = float(cbf[gm_mask].mean()) if gm_mask.any() else float("nan")
    median_gm_cbf = float(np.median(cbf[gm_mask])) if gm_mask.any() else float("nan")
    std_gm_cbf  = float(cbf[gm_mask].std())  if gm_mask.any() else float("nan")
    spatial_cov = (100.0 * std_gm_cbf / mean_gm_cbf) if mean_gm_cbf not in (0, float("nan")) else float("nan")

    # 4. Determine flags
    flags: list[str] = []
    t = thresholds

    if qei_result["qei"] < t["qei_min"]:
        flags.append(f"QEI={qei_result['qei']:.3f}<{t['qei_min']}")
    lo, hi = t["mean_gm_cbf_min"], t["mean_gm_cbf_max"]
    if lo < hi and not np.isnan(mean_gm_cbf):
        if mean_gm_cbf < lo:
            flags.append(f"meanGM_CBF={mean_gm_cbf:.1f}<{lo}")
        if mean_gm_cbf > hi:
            flags.append(f"meanGM_CBF={mean_gm_cbf:.1f}>{hi}")

    # Extract raw timeseries (mean whole brain signal for each volume)
    # ASL context tells us which volume is control vs label
    brain_mask = gm_mask | wm_mask | csf_mask
    raw_timeseries = []
    if brain_mask.any() and subject.asl_data is not None:
        for vol_idx in range(subject.asl_data.shape[-1]):
            vol = subject.asl_data[..., vol_idx]
            raw_timeseries.append(float(vol[brain_mask].mean()))
            
    return {
        "subject_id":    subject.subject_id,
        "session_id":    subject.session_id or "",
        "qei":           qei_result["qei"],
        "pss":           qei_result["pss"],
        "di":            qei_result["di"],
        "n_gm":          qei_result["n_gm"],
        "mean_gm_cbf":   round(mean_gm_cbf,   2),
        "median_gm_cbf": round(median_gm_cbf, 2),
        "std_gm_cbf":    round(std_gm_cbf,    2),
        "spatial_cov":   round(spatial_cov,   4),
        "n_volumes":     len(subject.asl_context),
        "raw_timeseries": raw_timeseries,
        "asl_context":   subject.asl_context,
        "gm_mask":       gm_mask,
        "wm_mask":       wm_mask,
        "csf_mask":      csf_mask,
        "flagged":       len(flags) > 0,
        "flags":         "; ".join(flags) if flags else "PASS",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    bids_root:  str | Path,
    output_dir: str | Path,
    subjects:   list[str] | None = None,
    thresholds: dict | None = None,
    save_plots: bool = True,
    live_html_path: str | Path | None = None,
    verbose:    bool = True,
) -> list[dict]:
    """
    Run the full QC pipeline over a BIDS ASL dataset.

    Parameters
    ----------
    bids_root  : path to the BIDS dataset root directory
    output_dir : where to write results (CSV + optional plots)
    subjects   : optional list of subject IDs to process
    thresholds : dict of threshold overrides; defaults to DEFAULT_THRESHOLDS
    save_plots : if True, save per-cohort QEI distribution plot
    live_html_path: if set, updates a standalone HTML dashboard at this path per subject
    verbose    : if True, print progress to stdout

    Returns
    -------
    list of per-subject result dicts (same rows as the output CSV)
    """
    bids_root  = Path(bids_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    results: list[dict] = []
    errors:  list[str]  = []

    if verbose:
        print(f"\n  Quality Check Toolbox — Real Data Pipeline")
        print(f"  Dataset  : {bids_root.resolve()}")
        print(f"  Output   : {output_dir.resolve()}")
        print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── For live UI ──
    from .live_html import generate_live_html
    import subprocess
    import shutil

    for i, subject in enumerate(iter_dataset(bids_root, subjects=subjects)):
        if verbose:
            print(f"  [{i+1:>3}] {subject.label:<25}", end=" ", flush=True)
        try:
            result = _qc_subject(subject, t)
            
            # Keep a reference to the CBF map for the live HTML plot
            result_copy = dict(result)
            result_copy["cbf_map"] = subject.cbf_map
            results.append(result_copy)
            
            if verbose:
                flag_str = "[!] FLAGGED" if result["flagged"] else "[OK] PASS"
                print(f"QEI={result['qei']:.3f}  {flag_str}")
                if result["flagged"]:
                    print(f"            -> {result['flags']}")
                    
            # ── Push Live Update ──────────────────────────────────────────────── #
            if os.environ.get("LIVE_HTML_WORKTREE_DIR"):
                wt_dir = os.environ.get("LIVE_HTML_WORKTREE_DIR")
                generate_live_html(results, total_subjects=0, output_path="live_index.html")
                shutil.copy("live_index.html", os.path.join(wt_dir, "index.html"))
                subprocess.run(["git", "-C", wt_dir, "add", "index.html"], check=True)
                subprocess.run(["git", "-C", wt_dir, "commit", "-m", f"QC update: {subject.label}"], check=False)
                subprocess.run(["git", "-C", wt_dir, "push", "origin", "HEAD:gh-pages"], check=False)
                
            if live_html_path:
                generate_live_html(results, total_subjects=0, output_path=str(live_html_path))
                
        except Exception as exc:  # noqa: BLE001
            msg = f"{subject.label}: {exc}"
            errors.append(msg)
            if verbose:
                print(f"ERROR — {exc}")

    # Remove the bulky cbf_map from results before CSV saving to avoid large memory / serialization bounds
    for r in results:
        r.pop("cbf_map", None)


    # ── Save CSV ──────────────────────────────────────────────────────────── #
    if results:
        csv_path = output_dir / "qc_results.csv"
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        if verbose:
            n_flagged = sum(r["flagged"] for r in results)
            print(f"\n  -----------------------------------------")
            print(f"  Processed : {len(results)} subjects")
            print(f"  Flagged   : {n_flagged} ({100*n_flagged/len(results):.0f}%)")
            if errors:
                print(f"  Errors    : {len(errors)}")
            print(f"  Results   -> {csv_path}")

        # ── Save summary plot ──────────────────────────────────────────────── #
        if save_plots:
            try:
                _save_summary_plot(results, output_dir, thresholds=t)
                if verbose:
                    print(f"  Plot      -> {output_dir / 'qc_summary.png'}")
            except Exception as exc:
                warnings.warn(f"Could not save summary plot: {exc}", stacklevel=2)

    if verbose:
        print()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Summary plot
# ──────────────────────────────────────────────────────────────────────────────

def _save_summary_plot(
    results: list[dict],
    output_dir: Path,
    thresholds: dict[str, Any] | None = None,
) -> None:
    """
    Save a 4-panel QC summary figure:
      - QEI distribution (histogram with PASS/FAIL shading)
      - PSS distribution
      - Mean GM CBF distribution
      - Spatial CoV distribution
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = thresholds or DEFAULT_THRESHOLDS
    qei_thr = th.get("qei_min", 0.30)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "ASL QC Summary — Dataset Distribution",
        fontsize=14, fontweight="bold", y=1.01,
    )

    metrics = [
        ("qei",         "QEI (pass/fail)",    "#4CAF50", qei_thr, True,  (0, 1)),
        ("pss",         "PSS (in QEI)",       "#2196F3", None,    True,  (0, 1)),
        ("mean_gm_cbf", "Mean GM map value",  "#FF9800", None,    False, None),
        ("spatial_cov", "Spatial CoV %",      "#9C27B0", None,    False, None),
    ]

    for ax, (key, title, color, threshold, higher_is_better, xlim) in zip(axes.flat, metrics):
        values = [r[key] for r in results if not isinstance(r[key], float) or not np.isnan(r[key])]
        if not values:
            ax.set_visible(False)
            continue

        arr = np.array(values)
        n_bins = min(30, max(10, len(arr) // 3))
        ax.hist(arr, bins=n_bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)

        if threshold is not None:
            ax.axvline(threshold, color="#F44336", linewidth=2, linestyle="--",
                       label=f"Threshold = {threshold}")
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            if higher_is_better:
                ax.axvspan(xmin, threshold, alpha=0.1, color="#F44336")
                ax.axvspan(threshold, xmax, alpha=0.1, color="#4CAF50")
            else:
                ax.axvspan(xmin, threshold, alpha=0.1, color="#4CAF50")
                ax.axvspan(threshold, xmax, alpha=0.1, color="#F44336")
            ax.legend(fontsize=9)

        if xlim:
            ax.set_xlim(xlim)

        mean_val = arr.mean()
        ax.axvline(mean_val, color="black", linewidth=1.5, linestyle=":",
                   label=f"Mean = {mean_val:.3f}")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(key.replace("_", " ").title())
        ax.set_ylabel("# Subjects")
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "qc_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m qc_toolbox.pipeline",
        description="Run the ASL QC pipeline on a BIDS dataset.",
    )
    p.add_argument("--bids",     required=True, help="Path to the BIDS dataset root")
    p.add_argument("--output",   required=True, help="Output directory for results")
    p.add_argument("--subjects", nargs="+", default=None, help="Subject IDs to process")
    p.add_argument("--no-plots", action="store_true",  help="Skip saving summary plots")
    p.add_argument(
        "--strict-qcbf",
        action="store_true",
        help="Use strict thresholds for quantified CBF maps (ml/100g/min), not raw difference maps",
    )
    p.add_argument(
        "--qei-min",  type=float, default=None,
        help="QEI pass threshold (overrides preset default)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    base = STRICT_QUANTIFIED_CBF_THRESHOLDS if args.strict_qcbf else DEFAULT_THRESHOLDS
    th = dict(base)
    if args.qei_min is not None:
        th["qei_min"] = args.qei_min
    run_pipeline(
        bids_root  = args.bids,
        output_dir = args.output,
        subjects   = args.subjects,
        thresholds = th,
        save_plots = not args.no_plots,
    )
