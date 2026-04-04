"""
Data-driven thresholds: 2-component GMM (PDF crossing between modes) vs Tukey IQR fences.

For a "higher is better" metric (e.g. QEI): GMM cut separates low vs high mode; flag below cut.
For "lower is better" (e.g. spatial CoV): flag above cut.

PSS/DI/n_gm are not derived here separately — they are inputs to QEI; cohort cuts focus on QEI
plus any independent scalar (e.g. spatial CoV).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import stats as scipy_stats
    from scipy.optimize import brentq
except ImportError:  # pragma: no cover
    brentq = None
    scipy_stats = None

try:
    from sklearn.mixture import GaussianMixture
except ImportError:  # pragma: no cover
    GaussianMixture = None  # type: ignore[misc, assignment]


@dataclass
class IQRResult:
    q1: float
    q3: float
    iqr: float
    lower: float
    upper: float
    n: int


@dataclass
class GMMThresholdResult:
    metric: str
    n: int
    higher_is_better: bool
    weight_good: float
    weight_poor: float
    mean_good: float
    mean_poor: float
    std_good: float
    std_poor: float
    threshold_crossing: float
    threshold_fallback_midpoint: float
    used_fallback: bool
    flag_below: bool


def iqr_bounds(values: np.ndarray, factor: float = 1.5) -> IQRResult:
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 4:
        raise ValueError(f"IQR needs ≥4 values, got {v.size}")
    q1, q3 = np.percentile(v, [25.0, 75.0])
    iqr = float(q3 - q1)
    return IQRResult(
        q1=float(q1),
        q3=float(q3),
        iqr=iqr,
        lower=float(q1 - factor * iqr),
        upper=float(q3 + factor * iqr),
        n=int(v.size),
    )


def _std_from_cov(cov: np.ndarray) -> float:
    c = np.asarray(cov, dtype=float)
    if c.ndim == 0:
        return float(np.sqrt(max(c, 1e-12)))
    if c.shape == (1, 1):
        return float(np.sqrt(max(c[0, 0], 1e-12)))
    return float(np.sqrt(max(float(c), 1e-12)))


def _two_gaussian_crossing(
    w0: float, m0: float, s0: float,
    w1: float, m1: float, s1: float,
    bracket_lo: float, bracket_hi: float,
) -> tuple[float, bool]:
    if brentq is None or scipy_stats is None:
        return 0.5 * (m0 + m1), True

    def diff(x: float) -> float:
        return (
            w0 * scipy_stats.norm.pdf(x, m0, s0)
            - w1 * scipy_stats.norm.pdf(x, m1, s1)
        )

    lo, hi = min(bracket_lo, bracket_hi), max(bracket_lo, bracket_hi)
    if lo >= hi:
        return 0.5 * (m0 + m1), True

    xs = np.linspace(lo, hi, 200)
    d = np.array([diff(float(x)) for x in xs])
    sign = np.sign(d)
    for i in range(len(xs) - 1):
        if sign[i] == 0:
            return float(xs[i]), False
        if sign[i] * sign[i + 1] < 0:
            try:
                return float(brentq(diff, float(xs[i]), float(xs[i + 1]), xtol=1e-6)), False
            except ValueError:
                break
    return 0.5 * (m0 + m1), True


def gmm_two_component_threshold(
    values: np.ndarray,
    metric: str = "metric",
    *,
    higher_is_better: bool = True,
    random_state: int = 0,
) -> GMMThresholdResult:
    if GaussianMixture is None:
        raise ImportError("pip install scikit-learn")

    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 10:
        raise ValueError(f"GMM needs ≥10 samples, got {v.size}")

    gmm = GaussianMixture(
        n_components=2, covariance_type="full", random_state=random_state, max_iter=200
    )
    gmm.fit(v.reshape(-1, 1))

    w = gmm.weights_.flatten()
    m = gmm.means_.flatten()
    cov = gmm.covariances_
    if cov.ndim == 3:
        s = np.array([_std_from_cov(cov[i]) for i in range(2)])
    else:
        s = np.sqrt(np.maximum(cov.flatten()[:2], 1e-12))

    order = np.argsort(m)
    if higher_is_better:
        idx_poor, idx_good = order[0], order[1]
    else:
        idx_good, idx_poor = order[0], order[1]

    wg, wp = float(w[idx_good]), float(w[idx_poor])
    mg, mp = float(m[idx_good]), float(m[idx_poor])
    sg, sp = float(s[idx_good]), float(s[idx_poor])

    t_cross, fallback = _two_gaussian_crossing(
        wg, mg, max(sg, 1e-6), wp, mp, max(sp, 1e-6), mg, mp,
    )

    return GMMThresholdResult(
        metric=metric,
        n=int(v.size),
        higher_is_better=higher_is_better,
        weight_good=wg,
        weight_poor=wp,
        mean_good=mg,
        mean_poor=mp,
        std_good=sg,
        std_poor=sp,
        threshold_crossing=t_cross,
        threshold_fallback_midpoint=0.5 * (mg + mp),
        used_fallback=fallback,
        flag_below=higher_is_better,
    )


def plot_metric_threshold_comparison(
    values: np.ndarray,
    metric_label: str,
    out_path: str | Path,
    *,
    higher_is_better: bool = True,
    random_state: int = 0,
    title: str | None = None,
) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if scipy_stats is None:
        raise ImportError("scipy required")

    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 10:
        raise ValueError("Need ≥10 samples")

    g = gmm_two_component_threshold(v, metric_label, higher_is_better=higher_is_better, random_state=random_state)
    iq = iqr_bounds(v)

    wg, wp, mg, mp = g.weight_good, g.weight_poor, g.mean_good, g.mean_poor
    sg, sp = max(g.std_good, 1e-9), max(g.std_poor, 1e-9)
    pad = 0.05 * (v.max() - v.min() + 1e-9)
    xs = np.linspace(v.min() - pad, v.max() + pad, 400)
    mix = wg * scipy_stats.norm.pdf(xs, mg, sg) + wp * scipy_stats.norm.pdf(xs, mp, sp)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.hist(v, bins=min(28, max(10, v.size // 3)), density=True, alpha=0.45, color="#4C72B0", edgecolor="white")
    ax.plot(xs, mix, color="#222", lw=2, label="Mixture")
    ax.plot(xs, wg * scipy_stats.norm.pdf(xs, mg, sg), "--", color="#2ca02c", lw=1.2, label=f"Good μ={mg:.3g}")
    ax.plot(xs, wp * scipy_stats.norm.pdf(xs, mp, sp), "--", color="#d62728", lw=1.2, label=f"Poor μ={mp:.3g}")
    ax.axvline(g.threshold_crossing, color="#d62728", lw=2.2, label=f"GMM cut = {g.threshold_crossing:.4g}")
    ax.axvline(iq.lower, color="#9467bd", ls=":", lw=1.3, label=f"IQR low {iq.lower:.4g}")
    ax.axvline(iq.upper, color="#9467bd", ls=":", lw=1.3, label=f"IQR high {iq.upper:.4g}")
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Density")
    ax.set_title(title or f"{metric_label} (n={v.size})")
    ax.legend(fontsize=8, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "gmm_cut": g.threshold_crossing,
        "iqr_low": iq.lower,
        "iqr_high": iq.upper,
        "n": int(v.size),
    }


def build_report_rows(
    triples: list[tuple[str, str, GMMThresholdResult, IQRResult]],
) -> list[dict[str, Any]]:
    rows = []
    for col, label, g, iq in triples:
        if g.higher_is_better:
            gmm_f = f"Flag if {label} < {g.threshold_crossing:.4g}"
            iqr_f = f"Flag if outside [{iq.lower:.4g}, {iq.upper:.4g}]"
        else:
            gmm_f = f"Flag if {label} > {g.threshold_crossing:.4g}"
            iqr_f = f"Flag if outside [{iq.lower:.4g}, {iq.upper:.4g}]"
        rows.append({
            "metric": col,
            "label": label,
            "n": g.n,
            "gmm_cut": g.threshold_crossing,
            "iqr_low": iq.lower,
            "iqr_high": iq.upper,
            "gmm_flag_rule": gmm_f,
            "iqr_flag_rule": iqr_f,
        })
    return rows


def write_report_md(rows: list[dict[str, Any]], path: str | Path) -> None:
    intro = (
        "# Thresholds: GMM vs IQR\n\n"
        "**GMM (2 Gaussians):** Fit two components; threshold = where their weighted PDFs cross "
        "(valley). One rule per metric: for QEI-like scores, flag *below* the cut; for noise-like "
        "metrics, flag *above*.\n\n"
        "**IQR:** Tukey fences — flag if value is below Q1−1.5×IQR or above Q3+1.5×IQR.\n\n"
        "PSS, DI, and negative-GM fraction are *not* listed here; they already enter **QEI**. "
        "Derive cohort cuts on **QEI** (and optionally **spatial CoV**, which is separate).\n\n"
    )
    table = [
        "| Metric | Label | n | GMM cut | IQR range | GMM rule | IQR rule |",
        "|--------|-------|---|---------|-----------|----------|----------|",
    ]
    for r in rows:
        table.append(
            f"| {r['metric']} | {r['label']} | {r['n']} | {r['gmm_cut']:.4g} | "
            f"[{r['iqr_low']:.4g}, {r['iqr_high']:.4g}] | {r['gmm_flag_rule']} | {r['iqr_flag_rule']} |"
        )
    Path(path).write_text(intro + "\n".join(table) + "\n", encoding="utf-8")


def write_report_json(payload: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
