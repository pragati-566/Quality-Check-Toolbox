"""
qei.py — Quality Evaluation Index for ASL CBF maps.

Reference:
    Dolui et al. (2024). Automated Quality Evaluation Index for Arterial Spin
    Labeling Derived Cerebral Blood Flow Maps. JMRI. doi:10.1002/jmri.29308

Formula:
    QEI = ∛( (1 - exp(-3·pss^2.4))  ·  exp(-(0.1·DI^0.9 + 2.8·nGMCBF^0.5)) )

Components:
    pss     — Pearson correlation between CBF map and structural pseudo-CBF
    DI      — Index of dispersion (variance/mean normalised by mean GM CBF)
    nGMCBF  — Fraction of GM voxels with negative CBF values
"""

import numpy as np


def _structural_similarity(cbf_map: np.ndarray,
                            gm_prob: np.ndarray,
                            wm_prob: np.ndarray,
                            brain_mask: np.ndarray) -> float:
    """
    Compute structural similarity (pss).

    The pseudo-structural CBF (spCBF) is a weighted combination of tissue
    probability maps that approximates the expected regional CBF pattern:
    higher CBF in GM than WM.  pss is the Pearson correlation between the
    actual CBF map and spCBF, evaluated within the brain mask.

    Parameters
    ----------
    cbf_map    : 3-D array of CBF values (ml/100g/min)
    gm_prob    : 3-D array of grey-matter probability [0, 1]
    wm_prob    : 3-D array of white-matter probability [0, 1]
    brain_mask : Boolean 3-D mask (True = voxel inside brain)

    Returns
    -------
    pss : float in [-1, 1]  (clamped to [0, 1] after for formula stability)
    """
    # Build pseudo-structural CBF: GM weighted higher than WM
    sp_cbf = 0.6 * gm_prob + 0.4 * wm_prob

    cbf_brain = cbf_map[brain_mask]
    sp_brain  = sp_cbf[brain_mask]

    if cbf_brain.std() == 0 or sp_brain.std() == 0:
        return 0.0

    pss = float(np.corrcoef(cbf_brain, sp_brain)[0, 1])
    # Clamp to [0, 1] — negative correlation means poor quality (treat as 0)
    return max(pss, 0.0)


def _index_of_dispersion(cbf_map: np.ndarray,
                          gm_mask: np.ndarray,
                          wm_mask: np.ndarray,
                          csf_mask: np.ndarray) -> float:
    """
    Compute the Index of Dispersion (DI).

    DI = Var(pooled CBF) / (Mean(pooled CBF) * Mean_GM_CBF)

    where 'pooled CBF' is the union of CBF values across GM, WM, and CSF
    voxels.  Normalising by mean GM CBF makes DI scale-invariant.

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
    pooled = np.concatenate([
        cbf_map[gm_mask],
        cbf_map[wm_mask],
        cbf_map[csf_mask],
    ])

    mean_pooled = pooled.mean()
    mean_gm     = cbf_map[gm_mask].mean()

    if mean_pooled == 0 or mean_gm == 0:
        return 0.0

    di = pooled.var() / (mean_pooled * mean_gm)
    return float(max(di, 0.0))


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


def compute_qei(cbf_map: np.ndarray,
                gm_mask: np.ndarray,
                wm_mask: np.ndarray,
                csf_mask: np.ndarray,
                gm_prob: np.ndarray | None = None,
                wm_prob: np.ndarray | None = None) -> dict:
    """
    Compute the Quality Evaluation Index (QEI) for an ASL CBF map.

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

    brain_mask = gm_mask | wm_mask | csf_mask

    pss    = _structural_similarity(cbf_map, gm_prob, wm_prob, brain_mask)
    di     = _index_of_dispersion(cbf_map, gm_mask, wm_mask, csf_mask)
    n_gm   = _negative_gm_fraction(cbf_map, gm_mask)

    # QEI formula
    term1  = 1.0 - np.exp(-3.0 * pss ** 2.4)
    term2  = np.exp(-(0.1 * di ** 0.9 + 2.8 * n_gm ** 0.5))
    qei    = float((term1 * term2) ** (1.0 / 3.0))

    return {
        "qei":  round(qei,  4),
        "pss":  round(pss,  4),
        "di":   round(di,   4),
        "n_gm": round(n_gm, 4),
    }
