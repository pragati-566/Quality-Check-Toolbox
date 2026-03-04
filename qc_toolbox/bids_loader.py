"""
bids_loader.py — BIDS-format ASL data loader for the QC Toolbox.

Supports the BIDS 1.5+ ASL specification:
  sub-<label>/
    perf/
      sub-<label>[_ses-<label>]_asl.nii[.gz]
      sub-<label>[_ses-<label>]_aslcontext.tsv
      sub-<label>[_ses-<label>]_m0scan.nii[.gz]   (optional)
      sub-<label>[_ses-<label>]_asl.json           (optional)

References:
  BIDS ASL Extension: https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#arterial-spin-labeling-perfusion-data
"""

from __future__ import annotations

import json
import os
import glob
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    raise ImportError(
        "nibabel is required for BIDS loading.\n"
        "Install with: pip install nibabel"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────

class ASLSubject:
    """
    Container for one subject's ASL data loaded from a BIDS dataset.

    Attributes
    ----------
    subject_id  : str          e.g. "sub-01"
    session_id  : str | None   e.g. "ses-baseline", or None
    cbf_map     : np.ndarray   shape (X, Y, Z), ml/100g/min or raw difference
    m0_map      : np.ndarray | None
    asl_context : list[str]    e.g. ["control", "label", "control", ...]
    metadata    : dict         parsed from _asl.json sidecar (may be {})
    affine      : np.ndarray   4×4 voxel-to-world matrix from the NIfTI header
    """

    def __init__(
        self,
        subject_id: str,
        session_id: str | None,
        cbf_map: np.ndarray,
        m0_map: np.ndarray | None,
        asl_context: list[str],
        metadata: dict,
        affine: np.ndarray,
    ):
        self.subject_id  = subject_id
        self.session_id  = session_id
        self.cbf_map     = cbf_map
        self.m0_map      = m0_map
        self.asl_context = asl_context
        self.metadata    = metadata
        self.affine      = affine

    @property
    def label(self) -> str:
        """Human-readable identifier (subject [+ session])."""
        if self.session_id:
            return f"{self.subject_id}/{self.session_id}"
        return self.subject_id

    def __repr__(self) -> str:
        shape = self.cbf_map.shape
        return (
            f"ASLSubject({self.label!r}, cbf_shape={shape}, "
            f"m0={'yes' if self.m0_map is not None else 'no'})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_perf_files(bids_root: Path, subject_id: str, session_id: str | None):
    """
    Return (asl_nii, context_tsv, m0_nii_or_None, metadata_or_{}) for one subject.
    Raises FileNotFoundError if the required ASL NIfTI is absent.
    """
    if session_id:
        perf_dir = bids_root / subject_id / session_id / "perf"
        prefix   = f"{subject_id}_{session_id}"
    else:
        perf_dir = bids_root / subject_id / "perf"
        prefix   = subject_id

    # Required: ASL 4-D NIfTI
    asl_candidates = list(perf_dir.glob(f"{prefix}_asl.nii*"))
    if not asl_candidates:
        raise FileNotFoundError(
            f"No ASL NIfTI found for {prefix} in {perf_dir}"
        )
    asl_nii = asl_candidates[0]

    # Required: aslcontext.tsv
    ctx_path = perf_dir / f"{prefix}_aslcontext.tsv"
    if not ctx_path.exists():
        warnings.warn(
            f"aslcontext.tsv not found for {prefix}; assuming control-label pairs.",
            stacklevel=2,
        )
        ctx_path = None

    # Optional: M0 scan
    m0_candidates = list(perf_dir.glob(f"{prefix}_m0scan.nii*"))
    m0_nii = m0_candidates[0] if m0_candidates else None

    # Optional: JSON sidecar
    json_path = perf_dir / f"{prefix}_asl.json"
    metadata  = {}
    if json_path.exists():
        with open(json_path) as fh:
            metadata = json.load(fh)

    return asl_nii, ctx_path, m0_nii, metadata


def _load_context(ctx_path: Path | None, n_volumes: int) -> list[str]:
    """
    Parse aslcontext.tsv and return a list of volume types.
    Falls back to alternating control/label if the file is absent.
    """
    if ctx_path is None:
        pattern = ["control", "label"]
        return [pattern[i % 2] for i in range(n_volumes)]

    rows = ctx_path.read_text().strip().splitlines()
    # First line is the header ("volume_type"), skip it
    if rows and rows[0].strip().lower() == "volume_type":
        rows = rows[1:]
    return [r.strip() for r in rows]


def _compute_mean_cbf(asl_data: np.ndarray, context: list[str]) -> np.ndarray:
    """
    Derive a simple mean CBF map from the raw 4-D ASL timeseries.

    Strategy:
    - If 'cbf' volumes exist in context (already processed), average them.
    - Otherwise compute pairwise (control - label) difference images and
      average those as a proxy for CBF (not quantified, but usable for QC).

    Returns a 3-D array (X, Y, Z).
    """
    ctx_arr = np.array(context)

    # Case 1: pre-computed CBF volumes
    cbf_indices = np.where(ctx_arr == "cbf")[0]
    if cbf_indices.size > 0:
        return asl_data[..., cbf_indices].mean(axis=-1)

    # Case 2: subtract label from control pair-by-pair
    control_idx = np.where(ctx_arr == "control")[0]
    label_idx   = np.where(ctx_arr == "label")[0]
    n_pairs = min(len(control_idx), len(label_idx))

    if n_pairs == 0:
        warnings.warn(
            "No control/label pairs found — returning mean of all volumes.",
            stacklevel=3,
        )
        return asl_data.mean(axis=-1)

    diffs = [
        asl_data[..., control_idx[i]] - asl_data[..., label_idx[i]]
        for i in range(n_pairs)
    ]
    return np.stack(diffs, axis=-1).mean(axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def discover_subjects(bids_root: str | Path) -> list[tuple[str, str | None]]:
    """
    Walk a BIDS dataset and return (subject_id, session_id) tuples for
    every subject that has a perf/ directory with an *_asl.nii* file.

    Parameters
    ----------
    bids_root : path to the root of the BIDS dataset

    Returns
    -------
    list of (subject_id, session_id) — session_id is None for single-session datasets
    """
    bids_root = Path(bids_root)
    found = []

    for sub_dir in sorted(bids_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        subject_id = sub_dir.name

        # Multi-session layout
        ses_dirs = sorted(sub_dir.glob("ses-*"))
        if ses_dirs:
            for ses_dir in ses_dirs:
                if ses_dir.is_dir() and list(
                    (ses_dir / "perf").glob("*_asl.nii*") if (ses_dir / "perf").is_dir() else []
                ):
                    found.append((subject_id, ses_dir.name))
        else:
            # Single-session layout
            perf_dir = sub_dir / "perf"
            if perf_dir.is_dir() and list(perf_dir.glob("*_asl.nii*")):
                found.append((subject_id, None))

    return found


def load_subject(
    bids_root: str | Path,
    subject_id: str,
    session_id: str | None = None,
) -> ASLSubject:
    """
    Load ASL data for a single BIDS subject.

    Parameters
    ----------
    bids_root  : path to BIDS dataset root
    subject_id : e.g. "sub-01"
    session_id : e.g. "ses-baseline", or None

    Returns
    -------
    ASLSubject instance
    """
    bids_root = Path(bids_root)
    asl_path, ctx_path, m0_path, metadata = _find_perf_files(
        bids_root, subject_id, session_id
    )

    # Load ASL image
    asl_img   = nib.load(str(asl_path))
    asl_data  = np.asarray(asl_img.dataobj, dtype=np.float32)
    affine    = asl_img.affine

    # Handle 3-D (single volume) vs. 4-D
    if asl_data.ndim == 3:
        asl_data = asl_data[..., np.newaxis]

    n_volumes = asl_data.shape[-1]
    context   = _load_context(ctx_path, n_volumes)

    cbf_map = _compute_mean_cbf(asl_data, context)

    # Load M0 if present
    m0_map = None
    if m0_path is not None:
        m0_img = nib.load(str(m0_path))
        m0_map = np.asarray(m0_img.dataobj, dtype=np.float32)

    return ASLSubject(
        subject_id  = subject_id,
        session_id  = session_id,
        cbf_map     = cbf_map,
        m0_map      = m0_map,
        asl_context = context,
        metadata    = metadata,
        affine      = affine,
    )


def iter_dataset(
    bids_root: str | Path,
    subjects: list[str] | None = None,
) -> Iterator[ASLSubject]:
    """
    Iterate over all valid ASL subjects in a BIDS dataset.

    Parameters
    ----------
    bids_root : path to BIDS dataset root
    subjects  : optional list of subject IDs to restrict to

    Yields
    ------
    ASLSubject (skips subjects where loading fails, emitting a warning)
    """
    all_subjects = discover_subjects(bids_root)

    if subjects is not None:
        filter_set = set(subjects)
        all_subjects = [(s, ses) for s, ses in all_subjects if s in filter_set]

    for subject_id, session_id in all_subjects:
        try:
            yield load_subject(bids_root, subject_id, session_id)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Skipping {subject_id}"
                + (f"/{session_id}" if session_id else "")
                + f": {exc}",
                stacklevel=2,
            )
