# Quality Check Toolbox v1.0

A Python demo of the **Quality Evaluation Index (QEI)** for Arterial Spin Labeling (ASL) Cerebral Blood Flow (CBF) maps.

> *Automated Quality Evaluation Index for Arterial Spin Labeling Derived Cerebral Blood Flow Maps.*
> Dolui et al., JMRI 2024. [doi:10.1002/jmri.29308](https://doi.org/10.1002/jmri.29308)

---

## What is ASL and why does QC matter?

**Arterial Spin Labeling (ASL)** is an MRI technique that measures cerebral blood flow (CBF) non-invasively by magnetically labelling water in blood as it enters the brain. It is widely used in studies of Alzheimer's disease, stroke, and other neurological conditions.

ASL CBF maps are however prone to:
- **Motion artefacts** — patient movement between label/control pairs
- **Low SNR** — only ~1% of the signal comes from labelled blood
- **Incomplete labelling** — when blood arrives late, CBF appears artificially low or negative
- **Noise** — poor shimming, RF inhomogeneity

Manual quality rating by radiologists is gold-standard but slow and subjective. **QEI automates this** with a single scalar score in [0, 1].

---

## QEI Formula

```
QEI = ∛( (1 - exp(-3·ρ_ss^2.4)) · exp(-(0.1·DI^0.9 + 2.8·p_nGMCBF^0.5)) )
```

| Paper symbol | Code variable | Component | What it catches |
|---|---|---|-----------------|
| **ρ_ss** | `pss` | Structural similarity | Pearson correlation between CBF and pseudo-structural CBF. Low = spatial pattern destroyed by noise/artefacts |
| **DI** | `di` | Index of dispersion | High variance across tissue = motion or incomplete labelling |
| **p_nGMCBF** | `n_gm` | Negative GM fraction | CBF is always positive physiologically — negatives are pure artefact |

QEI ranges from **0** (unusable) → **1** (excellent).

---

## Project Structure

```
Quality-Check-Toolbox/
├── qc_toolbox/
│   ├── qei.py         # Core QEI formula
│   ├── synthetic.py   # Synthetic brain + CBF map generator
│   └── visualize.py   # Plots & console report
├── demo.py            # Run the demo
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo (generates synthetic data + QEI report)
python demo.py
```

### What the demo does

1. Builds a synthetic 3D brain (64×64×30 voxels) with GM / WM / CSF regions
2. Generates three CBF maps — **Excellent**, **Average**, **Poor** — by adding increasing noise and artefacts
3. Computes QEI for each and prints a results table
4. Saves `qei_report.png` with CBF slices and a score bar chart

```
Case        QEI     pss     DI    neg GM   Grade
────────────────────────────────────────────────
Excellent  ~0.85   ~0.93   ~1.2    1.0%   Excellent
Average    ~0.60   ~0.72   ~4.5   10.0%   Average
Poor       ~0.20   ~0.40  ~18.0   35.0%   Poor
```

---

## Use with your own data

```python
from qc_toolbox.qei import compute_qei

# Load your arrays (e.g. from NIfTI via nibabel)
result = compute_qei(cbf_map, gm_mask, wm_mask, csf_mask)
print(result)
# {'qei': 0.74, 'pss': 0.85, 'di': 3.2, 'n_gm': 0.04}
```

---

## Learn more

| Resource | Link |
|----------|------|
| QEI paper (Dolui et al. 2024) | [doi:10.1002/jmri.29308](https://doi.org/10.1002/jmri.29308) |
| ASL MRI overview | [ISMRM ASL Perfusion Study Group](https://www.ismrm.org/study-groups/perfusion/) |
| ExploreASL QC toolbox | [ExploreASL on GitHub](https://github.com/ExploreASL/ExploreASL) |
