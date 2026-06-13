# circadian-mci-predictor

> **Circadian-Aware Mild Cognitive Impairment Predictor**  
> A novel two-stream deep learning framework using passive smartphone metadata and circadian phase estimation for 12-month MCI prediction — no wearables required.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Dataset: GLOBEM](https://img.shields.io/badge/Dataset-GLOBEM%20PhysioNet-green.svg)](https://physionet.org/content/globem/1.1/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Colab Ready](https://img.shields.io/badge/Colab-Ready-F9AB00.svg)](https://colab.research.google.com/)

---

## 🔬 Research Overview

Japan's population aged over 65 has reached 36 million, with dementia projected to affect 7 million by 2025. Existing AI approaches require structured wearables or active clinical testing. **This project uses only passive smartphone signals** — timestamps, screen-on events, app-switch patterns — combined with mathematical circadian rhythm modeling to predict mild cognitive impairment (MCI) 12 months before clinical diagnosis.

### What Makes This Novel

All prior passive sensing work uses either:
- Wearable hardware (accelerometers, EEG)
- Active cognitive test inputs
- Single behavioral signals without circadian modeling

**This project is the first to combine:**
1. Zero-hardware passive phone metadata (screen events, app patterns)
2. Mathematical circadian phase estimation via cosine curve fitting
3. A two-stream GRU + circadian MLP fusion architecture
4. Longitudinal MCI prediction with a 90-day prediction horizon

---

## 📂 Repository Structure

```
circadian-mci-predictor/
├── notebooks/
│   └── 01_circadian_mci_predictor.ipynb   ← Main research notebook (start here)
├── src/
│   ├── features/                           ← Feature extraction modules
│   ├── models/                             ← Model architectures
│   └── utils/                              ← Helper utilities
├── data/
│   ├── raw/          ← Place GLOBEM data here after download
│   ├── processed/    ← Preprocessed features
│   └── synthetic/    ← Auto-generated; run notebook to populate
├── results/
│   ├── figures/      ← Generated plots
│   └── *.json        ← Experiment results
├── docs/
├── tests/
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Option A: Google Colab (Recommended — no GPU needed for prototyping)

1. Upload `notebooks/01_circadian_mci_predictor.ipynb` to Colab
2. Run all cells — synthetic data is auto-generated
3. For real GLOBEM data: complete PhysioNet credentialing first (see below)

### Option B: Local CPU

```bash
git clone https://github.com/YOUR_USERNAME/circadian-mci-predictor.git
cd circadian-mci-predictor
pip install -r requirements.txt
jupyter notebook notebooks/01_circadian_mci_predictor.ipynb
```

---

## 📊 Dataset: GLOBEM (PhysioNet)

**GLOBEM** is the highest-quality public dataset for this research:

| Property | Value |
|---|---|
| Source | University of Washington |
| DOI | https://doi.org/10.13026/r9s1-s711 |
| Participants | 497 unique users |
| Data volume | 700+ user-years |
| Collection years | 2018–2021 (4 cohorts) |
| Modalities | Screen events, app usage, location, accelerometer, call/SMS |
| Labels | PHQ-9 depression / wellbeing (proxy for cognitive stress) |
| Access | Free — requires PhysioNet credentialing |

### How to Access GLOBEM

1. Register at https://physionet.org/register/
2. Complete CITI training module (~2 hours, free)
3. Sign data use agreement at https://physionet.org/content/globem/1.1/
4. Download:

```bash
wget -r -N -c -np \
  --user=YOUR_PHYSIONET_USERNAME \
  --ask-password \
  https://physionet.org/files/globem/1.1/ \
  -P data/raw/
```

5. Place downloaded folders inside `data/raw/`

> **Don't have access yet?** The notebook includes a high-fidelity synthetic dataset generator that mirrors GLOBEM's statistical properties. You can run all experiments immediately.

---

## 🧠 Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               CircadianMCIPredictor (~48K params)           │
│                                                             │
│  Stream A: Behavioral GRU                                   │
│  ┌───────────────────────────┐                              │
│  │  Raw + rolling features   │ → GRU (64 units, 2 layers)  │
│  │  (15 features × 30 days)  │ → h_A (64-dim)              │
│  └───────────────────────────┘                              │
│                                                             │
│  Stream B: Circadian Encoder  ← NOVEL COMPONENT            │
│  ┌───────────────────────────┐                              │
│  │  phase_est, amplitude,    │                              │
│  │  r², phase_std, CRS,      │ → MLP → h_B (32-dim)        │
│  │  drift_rate (7 features)  │                              │
│  └───────────────────────────┘                              │
│                                                             │
│  Fusion: [h_A ‖ h_B] → LayerNorm → Dropout → MLP → σ       │
└─────────────────────────────────────────────────────────────┘
```

### Circadian Regularity Score (CRS) — Core Novel Feature

$$\text{CRS}_t = 1 - \frac{\sigma(\hat{\phi}_{t-14:t})}{3 \cdot \sigma_{\text{healthy}}}$$

where $\hat{\phi}$ is the estimated circadian phase from cosine curve fitting to daily screen-on event distributions.

---

## 📈 Reproducibility Checklist

- [x] Synthetic dataset included (no credentials needed)
- [x] Fixed random seeds (42 everywhere)
- [x] Stratified 5-fold cross-validation
- [x] Ablation study: two-stream vs behavioral-only
- [x] Results saved as JSON
- [x] All figures saved to `results/figures/`

---

## 🗓️ Research Roadmap

| Phase | Timeline | Milestone |
|---|---|---|
| Foundation | Year 1 | GLOBEM experiments + AMIA/EMBC submission |
| Extension | Year 2 | ADNI validation + typing dynamics + Nature Digital Medicine |
| Translation | Year 3 | Prospective pilot + federated learning + thesis |

---

## 📖 Key References

1. Xu et al. (2023). GLOBEM Dataset: Multi-Year Datasets for Longitudinal Human Behavior Modeling Generalization. *NeurIPS Datasets & Benchmarks*. PhysioNet. https://doi.org/10.13026/r9s1-s711
2. Shimada et al. (2025). A New Computer-Based Cognitive Measure for Early Detection of Dementia Risk (Japan Cognitive Function Test). *JMIR*. https://doi.org/10.2196/59015
3. Ajilore et al. (2025). Assessment of cognitive function in bipolar disorder with passive smartphone keystroke metadata. *Frontiers in Psychiatry*. doi:10.3389/fpsyt.2025.1430303
4. Mahbub et al. (2026). Comprehensive review of AI as a catalyst in aging research. *Frontiers in Aging*. doi:10.3389/fragi.2026.1644669

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

*PhD Research Project | Medical AI — Cognitive Decline Prediction*