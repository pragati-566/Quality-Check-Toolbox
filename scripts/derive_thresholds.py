#!/usr/bin/env python3
"""
Derive cohort thresholds from qc_results.csv (GMM valley vs IQR fences).

Default metrics: QEI (composite) + spatial CoV (not inside QEI). Optional: --metrics pss di n_gm.

> python scripts/derive_thresholds.py --csv qc_output/qc_results.csv --output qc_output/threshold_analysis
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qc_toolbox.threshold_derivation import (  # noqa: E402
    GMMThresholdResult,
    IQRResult,
    build_report_rows,
    gmm_two_component_threshold,
    iqr_bounds,
    plot_metric_threshold_comparison,
    write_report_json,
    write_report_md,
)

# Default: QEI + one independent scalar. PSS/DI/n_gm are inside QEI — use --metrics to add them.
KNOWN = {
    "qei": ("QEI", True),
    "spatial_cov": ("Spatial CoV %", False),
    "pss": ("PSS", True),
    "di": ("DI", False),
    "n_gm": ("Negative GM fraction", False),
}
DEFAULT_KEYS = ("qei", "spatial_cov")


def load_column(csv_path: Path, column: str) -> np.ndarray:
    vals: list[float] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        if r.fieldnames is None or column not in r.fieldnames:
            raise KeyError(f"No column {column!r}")
        for row in r:
            try:
                x = float(row[column])
                if np.isfinite(x):
                    vals.append(x)
            except (TypeError, ValueError):
                continue
    return np.asarray(vals, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description="GMM + IQR thresholds from qc_results.csv")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--metrics",
        nargs="*",
        default=list(DEFAULT_KEYS),
        metavar="NAME",
        help=f"From: {', '.join(KNOWN)} (default: {' '.join(DEFAULT_KEYS)})",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    csv_path, out_dir = args.csv.resolve(), args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.is_file():
        print(f"Not found: {csv_path}", file=sys.stderr)
        return 1

    keys = args.metrics
    bad = set(keys) - set(KNOWN)
    if bad:
        print(f"Unknown metrics: {bad}", file=sys.stderr)
        return 1

    triples: list[tuple[str, str, GMMThresholdResult, IQRResult]] = []
    slim: dict[str, dict] = {}

    for key in keys:
        label, hib = KNOWN[key]
        try:
            v = load_column(csv_path, key)
        except KeyError as e:
            print(f"Skip {key}: {e}", file=sys.stderr)
            continue
        if v.size < 10:
            print(f"Skip {key}: n={v.size}", file=sys.stderr)
            continue
        try:
            g = gmm_two_component_threshold(v, key, higher_is_better=hib, random_state=args.seed)
            iq = iqr_bounds(v)
        except Exception as exc:  # noqa: BLE001
            print(f"Skip {key}: {exc}", file=sys.stderr)
            continue

        triples.append((key, label, g, iq))
        plot_metric_threshold_comparison(
            v, label, out_dir / f"{key}_gmm_iqr.png",
            higher_is_better=hib, random_state=args.seed,
        )
        slim[key] = {
            "label": label,
            "n": g.n,
            "gmm_cut": g.threshold_crossing,
            "iqr_low": iq.lower,
            "iqr_high": iq.upper,
            "flag_below_cut": g.flag_below,
        }

    if not triples:
        print("Nothing to write.", file=sys.stderr)
        return 1

    rows = build_report_rows(triples)
    write_report_md(rows, out_dir / "threshold_report.md")
    write_report_json({"source": str(csv_path), "seed": args.seed, "metrics": slim}, out_dir / "threshold_report.json")
    print(f"Wrote {out_dir}/ (figures, threshold_report.md, threshold_report.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
