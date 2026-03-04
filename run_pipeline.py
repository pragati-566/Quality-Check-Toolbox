#!/usr/bin/env python3
"""
run_pipeline.py — Command-line entry point for the QC Toolbox real-data pipeline.

Workflow:
  Run QC pipeline on local BIDS data

Examples
--------
# Run on ExploreASL test dataset or any local BIDS folder:
    python run_pipeline.py run --bids /path/to/my_bids_data --output ./qc_output

# Run with custom thresholds (e.g. pediatric population):
    python run_pipeline.py run --bids ./data/ExploreASL/External/TestDataSet/rawdata \
        --output ./qc_output \
        --qei-min 0.65 \
        --mean-gm-min 20 --mean-gm-max 80
"""

import argparse
import sys
import os

# Allow running from the project root without pip install
sys.path.insert(0, os.path.dirname(__file__))



def cmd_run(args):
    from qc_toolbox.pipeline import run_pipeline, DEFAULT_THRESHOLDS

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.qei_min   is not None: thresholds["qei_min"]         = args.qei_min
    if args.di_max    is not None: thresholds["di_max"]           = args.di_max
    if args.ngm_max   is not None: thresholds["n_gm_max"]         = args.ngm_max
    if args.mean_gm_min is not None: thresholds["mean_gm_cbf_min"] = args.mean_gm_min
    if args.mean_gm_max is not None: thresholds["mean_gm_cbf_max"] = args.mean_gm_max

    run_pipeline(
        bids_root  = args.bids,
        output_dir = args.output,
        subjects   = args.subjects,
        thresholds = thresholds,
        save_plots = not args.no_plots,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "run_pipeline.py",
        description = "Quality Check Toolbox — Real Data Pipeline",
        epilog      = "Use '<command> --help' for per-command options.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── run subcommand ──────────────────────────────────────────────────── #
    run = sub.add_parser(
        "run",
        help="Run the QC pipeline on a local BIDS dataset",
    )
    run.add_argument("--bids",        required=True,
                     help="Path to the BIDS dataset root")
    run.add_argument("--output",      required=True,
                     help="Directory to save QC results (CSV + plots)")
    run.add_argument("--subjects",    nargs="+", default=None,
                     help="Subject IDs to process (default: all)")
    run.add_argument("--no-plots",    action="store_true",
                     help="Skip saving summary plots")
    # Threshold overrides
    run.add_argument("--qei-min",     type=float, default=None,
                     metavar="FLOAT", help="Minimum QEI threshold (default 0.70)")
    run.add_argument("--di-max",      type=float, default=None,
                     metavar="FLOAT", help="Maximum DI threshold (default 2.00)")
    run.add_argument("--ngm-max",     type=float, default=None,
                     metavar="FLOAT", help="Max negative GM fraction (default 0.10)")
    run.add_argument("--mean-gm-min", type=float, default=None,
                     metavar="FLOAT", help="Min mean GM CBF ml/100g/min (default 10)")
    run.add_argument("--mean-gm-max", type=float, default=None,
                     metavar="FLOAT", help="Max mean GM CBF ml/100g/min (default 120)")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
