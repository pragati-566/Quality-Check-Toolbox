"""
tissue_masks.py — Derive GM / WM / CSF masks from real NIfTI CBF maps.

For real data without a separate segmentation available (e.g., when running
on raw OpenNeuro downloads without FreeSurfer/FSL outputs), we use intensity-
based heuristics on the CBF map itself to approximate tissue compartments.

If you have proper segmentation masks (from FreeSurfer, FSL FAST, or
SPM12), you should load those directly and pass them to compute_qei().

Strategy (CBF-based approximation)
------------------------------------
1. Brain mask  : voxels with |CBF| > noise_threshold (Otsu or fixed)
2. GM mask     : CBF in the upper range within brain (perfusion is higher in GM)
3. WM mask     : CBF in the lower-positive range within brain
4. CSF mask    : CBF near zero or negative (within brain boundary)

This is intentionally simple — for publication-quality work, replace with
tissue_masks_from_seg() using proper T1-derived segmentation.
"""

from __future__ import annotations

import warnings
import numpy as np

try:
    from scipy.ndimage import binary_fill_holes, binary_erosion, label as ndlabel
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def masks_from_cbf(
    cbf_map: np.ndarray,
    *,
    gm_percentile_range: tuple[float, float] = (60.0, 100.0),
    wm_percentile_range: tuple[float, float] = (20.0, 60.0),
    csf_percentile_range: tuple[float, float] = (0.0, 20.0),
    brain_threshold_pct: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Derive approximate GM / WM / CSF binary masks from a CBF map alone.

    This is a heuristic approach suitable for exploratory QC when tissue
    segmentation files are unavailable.  Masks are based on CBF intensity
    percentile bins within the detected brain region.

    Parameters
    ----------
    cbf_map : np.ndarray
        3-D CBF array (any units — values just need to be relatively higher
        in GM than WM).
    gm_percentile_range : (low_pct, high_pct)
        Percentile range within brain voxels assigned to GM.
    wm_percentile_range : (low_pct, high_pct)
        Percentile range within brain voxels assigned to WM.
    csf_percentile_range : (low_pct, high_pct)
        Percentile range within brain voxels assigned to CSF.
    brain_threshold_pct : float
        Voxels with |CBF| below this percentile of all absolute values are
        considered background (outside brain).

    Returns
    -------
    gm_mask, wm_mask, csf_mask : np.ndarray bool, same shape as cbf_map
    """
    abs_cbf = np.abs(cbf_map)
    threshold = np.percentile(abs_cbf, brain_threshold_pct)
    brain_mask = abs_cbf > threshold

    if not brain_mask.any():
        warnings.warn(
            "Brain mask is empty — check that the CBF map has non-zero values.",
            stacklevel=2,
        )
        empty = np.zeros(cbf_map.shape, dtype=bool)
        return empty, empty, empty

    brain_cbf = cbf_map[brain_mask]

    def pct_mask(low: float, high: float) -> np.ndarray:
        lo_val = np.percentile(brain_cbf, low)
        hi_val = np.percentile(brain_cbf, high)
        m = np.zeros(cbf_map.shape, dtype=bool)
        m[brain_mask] = (brain_cbf >= lo_val) & (brain_cbf <= hi_val)
        return m

    gm_mask  = pct_mask(*gm_percentile_range)
    wm_mask  = pct_mask(*wm_percentile_range)
    csf_mask = pct_mask(*csf_percentile_range)

    # Resolve overlaps: give priority GM > WM > CSF
    wm_mask  = wm_mask  & ~gm_mask
    csf_mask = csf_mask & ~gm_mask & ~wm_mask

    return gm_mask, wm_mask, csf_mask


def masks_from_segmentation(
    gm_seg: np.ndarray,
    wm_seg: np.ndarray,
    csf_seg: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert continuous tissue probability maps (e.g., from FSL FAST or SPM)
    into binary masks.

    Parameters
    ----------
    gm_seg, wm_seg, csf_seg : np.ndarray
        Probability maps in [0, 1], same shape.
    threshold : float
        Binarisation threshold (default 0.5).

    Returns
    -------
    gm_mask, wm_mask, csf_mask : np.ndarray bool
    """
    gm_mask  = gm_seg  > threshold
    wm_mask  = wm_seg  > threshold
    csf_mask = csf_seg > threshold
    return gm_mask, wm_mask, csf_mask
