"""
qei.py — Quality Evaluation Index for ASL CBF maps.

Reference:
    Dolui et al. (2024). Automated Quality Evaluation Index for Arterial Spin
    Labeling Derived Cerebral Blood Flow Maps. JMRI. doi:10.1002/jmri.29308

Formula (ASLPrep empirical coefficients):
    QEI = ∛( (1 - exp(α·pss^β)) · exp(-(γ·DI^δ + ε·nGMCBF^ζ)) )
    α=-3.0126, β=2.4419, γ=0.054, δ=0.9272, ε=2.8478, ζ=0.5196

Components:
    pss     — Pearson correlation between CBF map and structural pseudo-CBF
    DI      — Index of dispersion (within-tissue pooled variance / |mean GM CBF|)
    nGMCBF  — Fraction of GM voxels with negative CBF values
"""

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

# ASLPrep empirical coefficients
_QEI_ALPHA = -3.0126
_QEI_BETA = 2.4419
_QEI_GAMMA = 0.054
_QEI_DELTA = 0.9272
_QEI_EPSILON = 2.8478
_QEI_ZETA = 0.5196


def _structural_similarity(cbf_map: np.ndarray,
                            gm_prob: np.ndarray,
                            wm_prob: np.ndarray,
                            brain_mask: np.ndarray) -> float:
    """
    Compute structural similarity (pss).

    The pseudo-structural CBF (spCBF) = 2.5*GM + 1.0*WM reflects the expected
    GM:WM perfusion ratio.  pss is the Pearson correlation between the actual
    CBF map and spCBF, evaluated within valid brain voxels.

    Parameters
    ----------
    cbf_map    : 3-D array of CBF values (ml/100g/min)
    gm_prob    : 3-D array of grey-matter probability [0, 1]
    wm_prob    : 3-D array of white-matter probability [0, 1]
    brain_mask : Boolean 3-D mask (True = voxel inside brain)

    Returns
    -------
    pss : float in [0, 1]  (clamped; negative correlation → 0)
    """
    # Pseudo-structural CBF: 2.5*GM + 1.0*WM (referenced from ASLPrep)
    sp_cbf = 2.5 * gm_prob + 1.0 * wm_prob

    msk = (brain_mask
           & (cbf_map != 0)
           & ~np.isnan(cbf_map)
           & ~np.isnan(sp_cbf))
    cbf_vals = cbf_map[msk]
    sp_vals = sp_cbf[msk]

    if cbf_vals.size < 2 or cbf_vals.std() == 0 or sp_vals.std() == 0:
        return 0.0

    pss = float(np.corrcoef(cbf_vals, sp_vals)[0, 1])
    return max(pss, 0.0)


def _index_of_dispersion(cbf_map: np.ndarray,
                          gm_mask: np.ndarray,
                          wm_mask: np.ndarray,
                          csf_mask: np.ndarray) -> float:
    """
    Compute the Index of Dispersion (DI).

    V = ((n_gm-1)*var(gm) + (n_wm-1)*var(wm) + (n_csf-1)*var(csf)) / (n_gm+n_wm+n_csf-3)
    DI = V / |mean_gm_cbf|

    Parameters
    ----------
    cbf_map  : 3-D array of CBF values
    gm_mask  : Boolean 3-D mask for grey matter
    wm_mask  : Boolean 3-D mask for white matter
    csf_mask : Boolean 3-D mask for CSF

    Returns
    -------
    DI : float >= 0
    """
    n_gm = int(np.sum(gm_mask))
    n_wm = int(np.sum(wm_mask))
    n_csf = int(np.sum(csf_mask))

    if n_gm <= 1 or n_wm <= 1 or n_csf <= 1:
        raise ValueError(
            f"Insufficient voxels for variance computation: "
            f"GM={n_gm}, WM={n_wm}, CSF={n_csf}. Each must be > 1."
        )

    V = (
        (n_gm - 1) * np.var(cbf_map[gm_mask])
        + (n_wm - 1) * np.var(cbf_map[wm_mask])
        + (n_csf - 1) * np.var(cbf_map[csf_mask])
    ) / (n_gm + n_wm + n_csf - 3)

    mean_gm_cbf = np.mean(cbf_map[gm_mask])
    if np.isclose(mean_gm_cbf, 0):
        raise ValueError("Mean GM CBF is too close to zero, cannot compute dispersion index.")

    return float(max(V / np.abs(mean_gm_cbf), 0.0))


def _negative_gm_fraction(cbf_map: np.ndarray,
                           gm_mask: np.ndarray) -> float:
    """
    Fraction of grey-matter voxels with negative CBF values (nGMCBF).

    Physiological CBF is always positive; negative values are artefacts.

    Parameters
    ----------
    cbf_map : 3-D array of CBF values
    gm_mask : Boolean 3-D mask for grey matter

    Returns
    -------
    nGMCBF : float in [0, 1]
    """
    gm_cbf = cbf_map[gm_mask]
    if gm_cbf.size == 0:
        return 0.0
    return float((gm_cbf < 0).mean())


def _smooth_cbf(cbf_map: np.ndarray, fwhm_mm: float = 5.0,
                 voxel_dims_mm: tuple[float, float, float] | None = None) -> np.ndarray:
    """Apply Gaussian smoothing (FWHM) to CBF map. Uses voxel dims if provided."""
    if gaussian_filter is None:
        return cbf_map
    if voxel_dims_mm is None:
        voxel_dims_mm = (3.0, 3.0, 3.0)
    sigma_mm = fwhm_mm / (2 * np.sqrt(2 * np.log(2)))
    sigma_vox = [sigma_mm / d for d in voxel_dims_mm]
    return gaussian_filter(cbf_map.astype(np.float64), sigma=sigma_vox, mode="nearest")


def compute_qei(cbf_map: np.ndarray,
                gm_mask: np.ndarray,
                wm_mask: np.ndarray,
                csf_mask: np.ndarray,
                gm_prob: np.ndarray | None = None,
                wm_prob: np.ndarray | None = None,
                affine: np.ndarray | None = None) -> dict:
    """
    Compute the Quality Evaluation Index (QEI) for an ASL CBF map.

    CBF is smoothed with 5 mm FWHM before computing components.

    Parameters
    ----------
    cbf_map  : np.ndarray, shape (X, Y, Z)
        Quantitative CBF map in ml/100g/min.
    gm_mask  : np.ndarray bool, shape (X, Y, Z)
        Binary grey-matter mask.
    wm_mask  : np.ndarray bool, shape (X, Y, Z)
        Binary white-matter mask.
    csf_mask : np.ndarray bool, shape (X, Y, Z)
        Binary CSF mask.
    gm_prob  : np.ndarray float, optional
        Grey-matter probability map [0, 1].  If None, the binary mask is used.
    wm_prob  : np.ndarray float, optional
        White-matter probability map [0, 1].  If None, the binary mask is used.
    affine   : np.ndarray 4x4, optional
        Voxel-to-world matrix for voxel size derivation (smoothing). Default 3 mm isotropic.

    Returns
    -------
    dict with keys:
        qei     — final scalar score in [0, 1]
        pss     — structural similarity component
        di      — index of dispersion component
        n_gm    — negative GM CBF fraction component
    """
    if gm_prob is None:
        gm_prob = gm_mask.astype(float)
    if wm_prob is None:
        wm_prob = wm_mask.astype(float)

    voxel_dims = None
    if affine is not None and affine.shape >= (4, 4):
        voxel_dims = tuple(float(np.sqrt(np.sum(affine[:3, i] ** 2))) for i in range(3))
    smoothed = _smooth_cbf(cbf_map, fwhm_mm=5.0, voxel_dims_mm=voxel_dims)

    brain_mask = gm_mask | wm_mask | csf_mask
    pss    = _structural_similarity(smoothed, gm_prob, wm_prob, brain_mask)
    di     = _index_of_dispersion(smoothed, gm_mask, wm_mask, csf_mask)
    n_gm   = _negative_gm_fraction(smoothed, gm_mask)

    # QEI formula (ASLPrep empirical coefficients)
    term1 = 1.0 - np.exp(_QEI_ALPHA * pss ** _QEI_BETA)
    term2 = np.exp(-(_QEI_GAMMA * di ** _QEI_DELTA + _QEI_EPSILON * n_gm ** _QEI_ZETA))
    qei   = float((term1 * term2) ** (1.0 / 3.0))

    return {
        "qei":  round(qei,  4),
        "pss":  round(pss,  4),
        "di":   round(di,   4),
        "n_gm": round(n_gm, 4),
    }
