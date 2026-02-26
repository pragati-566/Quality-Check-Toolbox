"""
synthetic.py — Synthetic ASL/CBF test-data generator for the QC Toolbox demo.

Generates realistic-looking 3-D CBF maps and tissue masks at three quality
levels (excellent / average / poor) and saves them as NumPy `.npy` files so
the rest of the toolbox can load them without requiring nibabel.

Usage (from project root)
--------------------------
    python -m qc_toolbox.synthetic                    # saves to ./test_data/
    python -m qc_toolbox.synthetic --out my_dir       # custom output dir
    python -m qc_toolbox.synthetic --shape 96 96 40   # custom volume size

Output files (per quality level)
----------------------------------
    test_data/<quality>_cbf.npy      float32 CBF map  (ml/100g/min)
    test_data/<quality>_gm_prob.npy  float32 GM probability map [0,1]
    test_data/<quality>_wm_prob.npy  float32 WM probability map [0,1]
    test_data/gm_mask.npy            bool    shared grey-matter mask
    test_data/wm_mask.npy            bool    shared white-matter mask
    test_data/csf_mask.npy           bool    shared CSF mask

How it works
-------------
Step 1 — Tissue masks
    Three nested ellipsoids define GM (cortex), WM (fibres), and CSF
    (ventricles).  The z-axis is compressed slightly to match typical
    axial MRI geometry.

        r = sqrt(x^2 + y^2 + (1.6*z)^2)   (normalised coords in [-1,1])
        r < 0.90  →  brain (outer boundary)
        r < 0.65  →  WM + CSF
        r < 0.25  →  CSF

Step 2 — Base CBF map
    Voxels are seeded with physiologically plausible values:
        GM  ≈ 60 ml/100g/min
        WM  ≈ 25 ml/100g/min
        CSF ≈  5 ml/100g/min
    A Gaussian blur (σ=1.5) simulates MRI point-spread smoothing.

Step 3 — Quality degradation
    | Quality   | Noise σ | Neg. GM fraction | Artefact stripe |
    |-----------|---------|-----------------|-----------------:|
    | excellent |   3     |     1 %          |    none          |
    | average   |  12     |    10 %          |    mild          |
    | poor      |  28     |    35 %          |    strong        |

    Negative CBF voxels in GM simulate acquisition artefacts; they
    directly drive the `nGMCBF` term in the QEI formula.

Step 4 — Tissue probability maps
    Binary masks are Gaussian-blurred to produce smooth probability maps
    in [0,1], matching what tissue-segmentation tools (SPM, FSL FAST)
    would output.
"""

import argparse
import os

import numpy as np
from scipy.ndimage import gaussian_filter


# ─────────────────────────────────────────────────────────────────────────── #
# Tissue mask                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def make_tissue_masks(shape: tuple[int, int, int] = (64, 64, 30)):
    """
    Create binary GM, WM and CSF masks for a synthetic ellipsoidal brain.

    Parameters
    ----------
    shape : (X, Y, Z) voxel dimensions

    Returns
    -------
    gm_mask, wm_mask, csf_mask : bool ndarrays, each of shape `shape`
    """
    X, Y, Z = shape

    # Normalised coordinate grids in [-1, 1]
    x = np.linspace(-1, 1, X)
    y = np.linspace(-1, 1, Y)
    z = np.linspace(-1, 1, Z)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # Ellipsoidal distance — z compressed to approximate axial brain shape
    r = np.sqrt(xx**2 + yy**2 + (zz * 1.6)**2)

    outer = r < 0.90   # full brain
    inner = r < 0.65   # WM region
    core  = r < 0.25   # ventricles / CSF

    gm_mask  = outer & ~inner   # cortical shell
    wm_mask  = inner & ~core    # white-matter interior
    csf_mask = core             # central CSF

    return gm_mask, wm_mask, csf_mask


# ─────────────────────────────────────────────────────────────────────────── #
# CBF map                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def make_cbf_map(gm_mask: np.ndarray,
                 wm_mask: np.ndarray,
                 csf_mask: np.ndarray,
                 quality: str = "excellent",
                 seed: int = 42) -> dict:
    """
    Generate a synthetic CBF map with companion tissue-probability maps.

    Parameters
    ----------
    gm_mask, wm_mask, csf_mask : bool ndarrays (same shape)
    quality : one of {"excellent", "average", "poor"}
    seed    : random seed for reproducibility

    Returns
    -------
    dict with keys
        cbf_map : float32 ndarray — CBF values in ml/100g/min
        gm_prob : float32 ndarray — GM probability in [0, 1]
        wm_prob : float32 ndarray — WM probability in [0, 1]
    """
    quality = quality.lower()
    if quality not in ("excellent", "average", "poor"):
        raise ValueError(
            f"quality must be 'excellent', 'average', or 'poor'. Got '{quality}'."
        )

    rng   = np.random.default_rng(seed)
    shape = gm_mask.shape

    # ── Step 2: Base CBF map ────────────────────────────────────────────── #
    cbf = np.zeros(shape, dtype=np.float32)
    cbf[gm_mask]  = 60.0
    cbf[wm_mask]  = 25.0
    cbf[csf_mask] =  5.0
    cbf = gaussian_filter(cbf, sigma=1.5)   # MRI point-spread blur

    # ── Step 3: Quality-dependent degradation ───────────────────────────── #
    cfg = {
        "excellent": dict(noise_sigma=3.0,  neg_fraction=0.01, artefact_amp=0.0,  extra_smooth=1.0),
        "average":   dict(noise_sigma=12.0, neg_fraction=0.10, artefact_amp=8.0,  extra_smooth=0.0),
        "poor":      dict(noise_sigma=28.0, neg_fraction=0.35, artefact_amp=25.0, extra_smooth=0.0),
    }[quality]

    # Gaussian noise
    cbf += rng.normal(0, cfg["noise_sigma"], shape).astype(np.float32)

    # Structured artefact: bright axial stripe (simulates motion/ghosting)
    if cfg["artefact_amp"] > 0:
        stripe = np.zeros(shape, dtype=np.float32)
        z_mid  = shape[2] // 2
        stripe[:, :, z_mid - 2 : z_mid + 2] = cfg["artefact_amp"]
        cbf += stripe

    # Additional smoothing for excellent quality (more homogeneous)
    if cfg["extra_smooth"] > 0:
        cbf = gaussian_filter(cbf, sigma=cfg["extra_smooth"])

    # Force the target fraction of GM voxels negative (artefact simulation)
    gm_indices = np.argwhere(gm_mask)
    n_neg = int(len(gm_indices) * cfg["neg_fraction"])
    if n_neg > 0:
        chosen = rng.choice(len(gm_indices), size=n_neg, replace=False)
        for ix in gm_indices[chosen]:
            x, y, z = ix
            cbf[x, y, z] = -abs(float(cbf[x, y, z])) - 1.0

    # ── Step 4: Tissue probability maps ────────────────────────────────── #
    def _prob(mask):
        p  = gaussian_filter(mask.astype(np.float32), sigma=1.2)
        mx = p.max()
        return np.clip(p / mx if mx > 0 else p, 0.0, 1.0).astype(np.float32)

    return {
        "cbf_map": cbf,
        "gm_prob": _prob(gm_mask),
        "wm_prob": _prob(wm_mask),
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Save / load helpers                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

def save_dataset(out_dir: str, shape: tuple[int, int, int] = (64, 64, 30)):
    """
    Generate and save all three quality-level datasets to *out_dir*.

    Parameters
    ----------
    out_dir : directory path (created if it doesn't exist)
    shape   : voxel dimensions for the synthetic volume
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  Generating synthetic tissue masks  (shape={shape}) …")
    gm_mask, wm_mask, csf_mask = make_tissue_masks(shape)

    # Save shared masks once
    np.save(os.path.join(out_dir, "gm_mask.npy"),  gm_mask)
    np.save(os.path.join(out_dir, "wm_mask.npy"),  wm_mask)
    np.save(os.path.join(out_dir, "csf_mask.npy"), csf_mask)
    print(f"  ✔  Saved shared tissue masks → {out_dir}/{{gm,wm,csf}}_mask.npy")

    for quality, seed in [("excellent", 1), ("average", 2), ("poor", 3)]:
        print(f"\n  Generating '{quality}' CBF map …")
        data   = make_cbf_map(gm_mask, wm_mask, csf_mask, quality=quality, seed=seed)
        prefix = os.path.join(out_dir, quality)

        np.save(f"{prefix}_cbf.npy",     data["cbf_map"])
        np.save(f"{prefix}_gm_prob.npy", data["gm_prob"])
        np.save(f"{prefix}_wm_prob.npy", data["wm_prob"])

        gm_cbf  = data["cbf_map"][gm_mask]
        pct_neg = (gm_cbf < 0).mean() * 100
        mean_gm = gm_cbf[gm_cbf > 0].mean() if (gm_cbf > 0).any() else 0
        print(f"  ✔  Saved {quality}_{{cbf,gm_prob,wm_prob}}.npy")
        print(f"     mean GM CBF (positive voxels) = {mean_gm:.1f} ml/100g/min")
        print(f"     negative GM voxels            = {pct_neg:.1f} %")

    print(f"\n  All files written to: {os.path.abspath(out_dir)}\n")


def load_dataset(out_dir: str, quality: str) -> dict:
    """
    Load a previously saved synthetic dataset.

    Parameters
    ----------
    out_dir : directory written by save_dataset()
    quality : one of {"excellent", "average", "poor"}

    Returns
    -------
    dict with keys: cbf_map, gm_mask, wm_mask, csf_mask, gm_prob, wm_prob
    """
    p = os.path.join(out_dir, quality)
    return {
        "cbf_map":  np.load(f"{p}_cbf.npy"),
        "gm_prob":  np.load(f"{p}_gm_prob.npy"),
        "wm_prob":  np.load(f"{p}_wm_prob.npy"),
        "gm_mask":  np.load(os.path.join(out_dir, "gm_mask.npy")),
        "wm_mask":  np.load(os.path.join(out_dir, "wm_mask.npy")),
        "csf_mask": np.load(os.path.join(out_dir, "csf_mask.npy")),
    }


# ─────────────────────────────────────────────────────────────────────────── #
# CLI                                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic ASL/CBF test data for the QC Toolbox."
    )
    parser.add_argument(
        "--out", default="test_data",
        help="Output directory (default: ./test_data)",
    )
    parser.add_argument(
        "--shape", nargs=3, type=int, default=[64, 64, 30],
        metavar=("X", "Y", "Z"),
        help="Volume dimensions in voxels (default: 64 64 30)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    save_dataset(out_dir=args.out, shape=tuple(args.shape))
