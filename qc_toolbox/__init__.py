"""Quality Check Toolbox v1.0 — ASL MRI QC pipeline."""

# ── Core QC metric ────────────────────────────────────────────────────────── #
from .qei import compute_qei

# ── Synthetic data (demo / testing) ──────────────────────────────────────── #
try:
    from .synthetic import make_tissue_masks, make_cbf_map, save_dataset, load_dataset
except ImportError:
    pass  # synthetic module is optional — not available in all environments

# ── Real data pipeline ────────────────────────────────────────────────────── #
from .bids_loader  import discover_subjects, load_subject, iter_dataset, ASLSubject
from .tissue_masks import masks_from_cbf, masks_from_segmentation
from .pipeline     import (
    run_pipeline,
    DEFAULT_THRESHOLDS,
    STRICT_QUANTIFIED_CBF_THRESHOLDS,
)
from .threshold_derivation import (
    gmm_two_component_threshold,
    iqr_bounds,
    plot_metric_threshold_comparison,
    build_report_rows,
)

__all__ = [
    # QEI
    "compute_qei",
    # Synthetic
    "make_tissue_masks", "make_cbf_map", "save_dataset", "load_dataset",
    # BIDS
    "discover_subjects", "load_subject", "iter_dataset", "ASLSubject",
    # Tissue masks
    "masks_from_cbf", "masks_from_segmentation",
    # Pipeline
    "run_pipeline",
    "DEFAULT_THRESHOLDS",
    "STRICT_QUANTIFIED_CBF_THRESHOLDS",
    # Threshold derivation
    "gmm_two_component_threshold",
    "iqr_bounds",
    "plot_metric_threshold_comparison",
    "build_report_rows",
]
