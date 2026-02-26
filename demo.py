#!/usr/bin/env python3
"""
demo.py — Quality Check Toolbox v1.0: QEI Demo

Generates three synthetic ASL CBF maps (excellent / average / poor quality),
computes the Quality Evaluation Index for each, prints a results table,
and saves a visual report to qei_report.png.

Usage
-----
    python demo.py

Requirements
------------
    pip install numpy scipy matplotlib
"""

import sys
import os

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(__file__))

from qc_toolbox.synthetic import make_tissue_masks, make_cbf_map
from qc_toolbox.qei       import compute_qei
from qc_toolbox.visualize import print_report, plot_qei_report


def main():
    print("\n  Quality Check Toolbox v1.0 — QEI Demo")
    print("  Based on: Dolui et al. 2024, JMRI (doi:10.1002/jmri.29308)\n")

    # ── 1. Shared anatomy ────────────────────────────────────────────────── #
    SHAPE = (64, 64, 30)
    print("  [1/3]  Generating synthetic tissue masks …", end=" ", flush=True)
    gm_mask, wm_mask, csf_mask = make_tissue_masks(shape=SHAPE)
    print("done")

    # ── 2. Compute QEI for three quality levels ───────────────────────────  #
    results = []
    for quality, seed in [("excellent", 1), ("average", 2), ("poor", 3)]:
        print(f"  [2/3]  Simulating '{quality}' CBF map and computing QEI …",
              end=" ", flush=True)
        data = make_cbf_map(gm_mask, wm_mask, csf_mask,
                            quality=quality, seed=seed)

        metrics = compute_qei(
            cbf_map  = data["cbf_map"],
            gm_mask  = gm_mask,
            wm_mask  = wm_mask,
            csf_mask = csf_mask,
            gm_prob  = data["gm_prob"],
            wm_prob  = data["wm_prob"],
        )
        results.append({
            "label":   quality.capitalize(),
            "cbf_map": data["cbf_map"],
            **metrics,
        })
        print("done")

    # ── 3. Report ─────────────────────────────────────────────────────────  #
    print_report(results)

    print("  [3/3]  Saving visual report …", end=" ", flush=True)
    out_path = os.path.join(os.path.dirname(__file__), "qei_report.png")
    plot_qei_report(results, save_path=out_path)


if __name__ == "__main__":
    main()
