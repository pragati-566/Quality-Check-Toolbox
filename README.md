# Quality Check Toolbox v1.0

**Proposed mentors:** Maria Mora, Sudipto Dolui  
**Category:** Advanced · Python · GitHub · MRI Image Processing

---

## What is this?

A Python demo of the **Quality Evaluation Index (QEI)** for Arterial Spin Labeling (ASL) Cerebral Blood Flow (CBF) maps, as described in:

> Dolui S, Wang Z, Wolf RL, et al.  
> *Automated Quality Evaluation Index for Arterial Spin Labeling Derived Cerebral Blood Flow Maps.*  
> JMRI 2024. [doi:10.1002/jmri.29308](https://doi.org/10.1002/jmri.29308)

---

## QEI Formula

$$
\text{QEI} = \sqrt[3]{\;\bigl(1 - e^{-3\,p_{ss}^{2.4}}\bigr)\cdot e^{-(0.1\,\text{DI}^{0.9}\;+\;2.8\,n_{\text{GMCBF}}^{0.5})}\;}
$$

| Symbol | Component | Description |
|--------|-----------|-------------|
| **pss** | Structural similarity | Pearson correlation between CBF map and pseudo-structural CBF (weighted GM + WM probability maps) |
| **DI** | Index of dispersion | Variance/mean of pooled tissue CBF, normalised by mean GM CBF — measures spatial variability |
| **nGMCBF** | Negative GM fraction | Fraction of GM voxels with CBF < 0 (non-physiological artefacts) |

QEI ranges from **0** (unusable) to **1** (excellent quality).

---

## Project Structure

```
Quality-Check-Toolbox/
├── qc_toolbox/
│   ├── __init__.py
│   ├── qei.py            # Core QEI computation (all 3 formula components)
│   ├── synthetic_data.py # Synthetic brain mask + CBF map generator
│   └── visualize.py      # Plotting & console report
├── demo.py               # Run the demo
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone / open the project
cd Quality-Check-Toolbox

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo
python demo.py
```

### Expected output

```
  Quality Check Toolbox v1.0 — QEI Demo
  Based on: Dolui et al. 2024, JMRI (doi:10.1002/jmri.29308)

  ...

────────────────────────────────────────────────────────────
Case            QEI     pss      DI    neg GM  Grade
────────────────────────────────────────────────────────────
Excellent    0.8xxx  0.9xxx    x.xx    0.x%   Excellent
Average      0.6xxx  0.7xxx    x.xx   10.x%   Average
Poor         0.2xxx  0.4xxx   xx.xx   35.x%   Poor
────────────────────────────────────────────────────────────
```

A **`qei_report.png`** is also saved, showing mid-axial CBF slices and a bar chart of scores.
