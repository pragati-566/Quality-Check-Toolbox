# Quality Check Toolbox v1.0

A Python toolbox for **automated Quality Control (QC) of Arterial Spin Labeling (ASL) MRI data**, featuring a full real-data pipeline for BIDS-formatted datasets.

*This project is developed as part of a GSoC proposal for Quality Check Toolbox v1.0.*

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
│   ├── qei.py           # Core QEI formula (Dolui et al. 2024)
│   ├── visualize.py     # Plots & console report
│   ├── bids_loader.py   # BIDS-format ASL data loader
│   ├── tissue_masks.py  # Tissue mask derivation from real CBF maps
│   └── pipeline.py      # Main QC pipeline runner
├── run_pipeline.py      # CLI entry point for real-data pipeline
├── requirements.txt
└── README.md
```

---

## Real-Data Pipeline

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the ExploreASL TestDataSet

Clone the ExploreASL repository to get access to their `TestDataSet`:

```bash
git clone --depth 1 https://github.com/ExploreASL/ExploreASL data/ExploreASL
```

### 3. Run QC pipeline using ExploreASL TestDataSet

```bash
python run_pipeline.py run --bids ./data/ExploreASL/External/TestDataSet/rawdata --output ./qc_output
```

**Output:**
- `qc_output/qc_results.csv` — per-subject QEI, PSS, DI, nGM, mean CBF, flags
- `qc_output/qc_summary.png` — 4-panel distribution plot

### 4. Custom thresholds (optional)

```bash
# Pediatric or clinical population with different physiology
python run_pipeline.py run \
    --bids ./data/ExploreASL/External/TestDataSet/rawdata \
    --output ./qc_output \
    --qei-min 0.65 \
    --mean-gm-min 20 \
    --mean-gm-max 80
```

### 5. Run on your own BIDS data (skip download)

```bash
python run_pipeline.py run --bids /path/to/my_bids_dataset --output ./qc_output
```

---

## Default QC Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| QEI | ≥ 0.70 | Dolui et al. 2024 recommended pass threshold |
| PSS | ≥ 0.40 | Low structural similarity = spatial artefacts |
| DI | ≤ 2.00 | High DI = noise or motion dominance |
| neg GM fraction | ≤ 0.10 | >10% negative GM CBF = severe artefact |
| Mean GM CBF | 10–120 ml/100g/min | Physiological plausibility range |

---

## Learn more

| Resource | Link |
|----------|------|
| QEI paper (Dolui et al. 2024) | [doi:10.1002/jmri.29308](https://doi.org/10.1002/jmri.29308) |
| BIDS ASL specification | [BIDS ASL Extension](https://bids-specification.readthedocs.io) |
| OpenNeuro datasets | [openneuro.org](https://openneuro.org) |
| ExploreASL QC toolbox | [ExploreASL on GitHub](https://github.com/ExploreASL/ExploreASL) |
