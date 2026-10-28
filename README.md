# Topography-Adaptive Road Circuity and Robust Gradient Boosting for Multi-Carrier Freight Transportation ETA Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Dataset: 35k Commercial Waybills](https://img.shields.io/badge/Dataset-35%2C000%20Waybills%20(SHA--256)-success.svg)](data/)
[![Benchmark Standard](https://img.shields.io/badge/Benchmark-IEEE%20Reproducibility-brightgreen.svg)](results/)
[![Inference Latency](https://img.shields.io/badge/Inference%20Latency-p95%20%3D%200.90%20ms-orange.svg)](results/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Official open-source research replication package and benchmark dataset for the paper:
> **"Topography-Adaptive Road Circuity and Robust Gradient Boosting for Multi-Carrier Freight Transportation ETA Prediction"**  
> *Umer Bashir, Muhammad Usman Shoukat, Muhammad Usman Shams (Corresponding Author)*

---

## 📑 Table of Contents

- [Overview & Technical Abstract](#-overview--technical-abstract)
- [Key Research Contributions](#-key-research-contributions)
- [Mathematical Formulation](#-mathematical-formulation)
- [Data Provenance & Privacy Statement](#-data-provenance--privacy-statement)
- [Dataset Schema & Feature Definitions](#-dataset-schema--feature-definitions)
- [Empirical Benchmarking Results](#-empirical-benchmarking-results)
- [Repository Structure](#-repository-structure)
- [Step-by-Step Reproduction Guide](#-step-by-step-reproduction-guide)
- [Hardware & Environment Specifications](#-hardware--environment-specifications)
- [Citation](#-citation)
- [Reviewer Reproducibility Statement](#-reviewer-reproducibility-statement)

---

## 🔬 Overview & Technical Abstract

Accurate Estimated Time of Arrival (ETA) prediction across multi-carrier intercity linehaul freight remains challenging due to complex topography, roadway circuity variations, carrier velocity heterogeneity, and extreme linehaul delay spikes. Classical routing engines (such as standard OSRM or Haversine distance heuristics) suffer significant prediction errors because they fail to account for nonlinear road terrain curvature and mountain elevation gradients, while standard Mean Squared Error (MSE) models diverge in the presence of heavy-tailed distribution shocks.

This research introduces a **Topography-Adaptive Road Circuity Framework** paired with an outlier-resilient **Huber-Loss Gradient Boosted Decision Tree (GBDT)** architecture. Evaluated on an authentic transcontinental dataset of $N = 35,000$ commercial freight waybills across four tier-1 commercial carriers (**DHL Express**, **FedEx Ground**, **UPS Worldwide**, and **TCS Express**), our framework achieves:
- **0.51 hours Mean Absolute Error (MAE)** (a **42.7% reduction** over static OSRM routing engines).
- **0.86 hours Root Mean Squared Error (RMSE)** and **4.1% Mean Absolute Percentage Error (MAPE)**.
- **$p95$ single-item inference latency of $0.90\text{ ms}$**, scaling to **$15,120\text{ queries/second}$** on standard commodity hardware.

---

## 🎯 Key Research Contributions

1. **Topography-Adaptive Road Circuity ($\kappa(\Delta h, d)$):** A closed-form geospatial formulation modeling elevation gradient attenuation and roadway sinuosity, replacing static geometric circuity assumptions.
2. **Outlier-Resilient Huber Optimization:** Calibration of Huber loss threshold ($\delta = 1.35$) providing optimal balance between $L_2$ gradient sensitivity for normal transit and $L_1$ linear robustness against multi-hour congestion outliers.
3. **Multi-Carrier Empirical Validation:** Transcontinental evaluation spanning $35,000$ real-world waybills across North American, Western European, and South Asian linehaul corridors.
4. **Sub-Millisecond Industrial Latency:** Algorithmic optimization delivering $p95 = 0.90\text{ ms}$ query latency, making it directly deployable within high-throughput Transportation Management Systems (TMS).

---

## 📐 Mathematical Formulation

### 1. Topography-Adaptive Circuity Factor
The effective highway path distance $d_{\text{eff}}$ between dispatch facility $i$ and destination facility $j$ is formulated as:
$$\kappa(\Delta h, d) = \kappa_0 \cdot \left(1 + \beta \frac{|\Delta h_{ij}|}{d_{ij} + \epsilon}\right)^\gamma$$
$$d_{\text{eff}} = \kappa(\Delta h, d) \cdot d_{\text{Haversine}}(i, j)$$
Where:
- $\kappa_0 = 1.27$ is the baseline planar circuity constant.
- $\Delta h_{ij} = h_j - h_i$ is the digital elevation differential between facilities in meters.
- $\beta = 0.42$ is the terrain slope coupling coefficient.
- $\gamma = 0.88$ is the non-linear gradient attenuation exponent.

### 2. Outlier-Robust Huber Objective Function
To ensure resilience against non-Gaussian delivery delays (border inspections, winter storm blockages, mechanical breakdowns), model optimization minimizes:
$$L_\delta(y, \hat{y}) = \begin{cases} 
\frac{1}{2} (y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\
\delta \cdot (|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}$$
With empirical threshold parameter $\delta = 1.35\text{ hours}$.

---

## 🛡 Data Provenance & Privacy Statement

> **Authentic Data Statement (Section IV-A):**  
> *The experimental dataset comprises $N = 35,000$ commercial freight waybills spanning transcontinental linehaul corridors across North America, Western Europe, and South Asia. The data were extracted from an enterprise Transportation Management System (TMS) archive and historical EDI telemetry streams across four commercial carriers (DHL Express, FedEx Ground, UPS Worldwide, and TCS Express). To protect personal data and commercial confidentiality, shipper/consignee identities and account identifiers were anonymized using SHA-256 cryptographic hashing, and facility coordinates were spatially perturbed by $\pm 0.005^\circ$, thereby strictly preserving authentic road-network topology, digital elevation profiles, dispatch timestamps, and observed transit durations while preventing reverse-geocoding.*

- **CSV File:** [`data/freight_waybills_35k_anonymized.csv`](data/freight_waybills_35k_anonymized.csv) (6.62 MB, 35,000 rows × 19 columns)
- **Compressed Archive:** [`data/freight_waybills_35k_anonymized.zip`](data/freight_waybills_35k_anonymized.zip) (1.69 MB)
- **SHA-256 Checksum (CSV):** `a981c2f10b77e8e4a9194ec710ec4d6189bb5fa5f272a81387d890bfca5cc24a`

---

## 📊 Dataset Schema & Feature Definitions

The dataset contains 19 structured features capturing spatial, temporal, topographic, and carrier telemetry attributes:

| # | Feature Column | Data Type | Units / Range | Description |
|---|---|---|---|---|
| 1 | `waybill_id_hash` | String | SHA-256 (64 hex) | Cryptographically pseudonymized waybill identifier |
| 2 | `carrier` | Categorical | DHL, FedEx, UPS, TCS | Commercial linehaul freight carrier |
| 3 | `origin_lat` | Float | $-90.0^\circ$ to $+90.0^\circ$ | Perturbed dispatch facility latitude ($\pm 0.005^\circ$) |
| 4 | `origin_lon` | Float | $-180.0^\circ$ to $+180.0^\circ$ | Perturbed dispatch facility longitude ($\pm 0.005^\circ$) |
| 5 | `dest_lat` | Float | $-90.0^\circ$ to $+90.0^\circ$ | Perturbed delivery facility latitude ($\pm 0.005^\circ$) |
| 6 | `dest_lon` | Float | $-180.0^\circ$ to $+180.0^\circ$ | Perturbed delivery facility longitude ($\pm 0.005^\circ$) |
| 7 | `origin_elevation_m` | Float | Meters ($0$ to $2,850$) | Digital elevation model (DEM) elevation at origin |
| 8 | `dest_elevation_m` | Float | Meters ($0$ to $2,850$) | Digital elevation model (DEM) elevation at destination |
| 9 | `elevation_diff_m` | Float | Meters | Net topographic elevation differential $|\Delta h|$ |
| 10 | `haversine_distance_km`| Float | Kilometers ($45$ to $3,200$) | Great-circle geodesic distance between origin and dest |
| 11 | `route_circuity_factor`| Float | Unitless ($1.15$ to $1.82$) | Calibrated terrain circuity factor $\kappa(\Delta h, d)$ |
| 12 | `service_level` | Categorical | Express, Ground, Freight | Commercial delivery tier |
| 13 | `dispatch_hour` | Integer | $0$ to $23$ | Hour of linehaul departure (local facility time) |
| 14 | `dispatch_dow` | Integer | $0$ to $6$ | Day of week of departure ($0=\text{Monday}$, $6=\text{Sunday}$) |
| 15 | `linehaul_dwell_hours` | Float | Hours ($0.2$ to $14.5$) | Intermediate cross-docking and sorting dwell time |
| 16 | `carrier_reliability_score`| Float | $0.00$ to $1.00$ | Historical 30-day on-time performance index for carrier |
| 17 | `weather_severity_index` | Float | $0.00$ to $1.00$ | Corridor meteorological adversity coefficient |
| 18 | `actual_transit_hours` | Float | Hours | Ground-truth observed transit duration from TMS |
| 19 | `huber_eta_prediction_hours` | Float | Hours | Topography-adaptive Huber GBDT prediction |

---

## 🏆 Empirical Benchmarking Results

### Table I: Transcontinental ETA Model Comparison ($N = 35,000$ Waybills)

All models evaluated on identical $80/20$ train/test splits with 5-fold cross-validation:

| Model Architecture | MAE (h) ↓ | RMSE (h) ↓ | MAPE (%) ↓ | $R^2$ Score ↑ | $p95$ Latency (ms) ↓ |
|---|---|---|---|---|---|
| **Proposed Huber GBDT ($\delta=1.35$)** | **0.51** | **0.86** | **4.1%** | **0.962** | **0.90** |
| LightGBM Regressor | 0.54 | 0.91 | 4.4% | 0.957 | 1.15 |
| CatBoost Regressor | 0.53 | 0.89 | 4.3% | 0.959 | 2.40 |
| XGBoost Regressor | 0.56 | 0.94 | 4.6% | 0.953 | 1.85 |
| Random Forest Regressor (100 trees) | 0.68 | 1.14 | 5.8% | 0.931 | 8.20 |
| Deep Multi-Layer Perceptron (MLP) | 0.74 | 1.28 | 6.4% | 0.914 | 3.50 |
| Static OSRM Routing Baseline | 0.89 | 1.52 | 8.2% | 0.862 | 22.40 |
| Linear Regression (OLS) | 1.18 | 1.84 | 10.5% | 0.798 | 0.15 |

### Key Experimental Insights:
- **Circuity Contribution:** Factoring in $\kappa(\Delta h, d)$ alone accounts for a **$0.21\text{ h}$ drop in MAE** across mountainous transit corridors.
- **Huber Robustness:** Under synthetic and observed shock testing ($15\%$ multi-hour linehaul delays), Huber loss maintained $0.51\text{ h MAE}$ where MSE-trained models degraded to $> 0.79\text{ h MAE}$.
- **SLA Compliance:** Within a $\pm 1.0\text{ h}$ arrival compliance window, the proposed architecture achieves **$94.8\%$ on-time accuracy**.

---

## 📁 Repository Structure

```
Paper1/
├── data/
│   ├── freight_waybills_35k_anonymized.csv   # Primary benchmark dataset (35,000 rows)
│   └── freight_waybills_35k_anonymized.zip   # Compressed archive
├── figures/                                  # Publication-ready vector & raster figures
│   ├── fig1_introduction.png                 # Fig. 1: Logistics ETA Problem Formulation
│   ├── fig2_background.png                   # Fig. 2: Road Circuity & Topography
│   ├── fig3_methodology.png                  # Fig. 3: Proposed Architecture Pipeline
│   ├── fig4_actual_vs_predicted_hexbin.png   # Fig. 4: Actual vs Predicted Hexbin Density
│   ├── fig5_residual_distribution_bias.png   # Fig. 5: Error Residual Distributions
│   ├── fig6_baseline_model_comparison_mae.png# Fig. 6: Baseline Model MAE Comparison
│   ├── fig7_compliance_window_accuracy.png   # Fig. 7: Compliance Window SLA Accuracy
│   ├── fig8_feature_importance_mdi_permutation.png # Fig. 8: MDI & Permutation Feature Importance
│   ├── fig9_corridor_distance_ablation.png   # Fig. 9: Corridor Distance Ablation Study
│   ├── fig10_huber_loss_convergence_and_latency.png # Fig. 10: Convergence & p95 Latency
│   ├── fig11_spatiotemporal_zone_and_dow_error.png # Fig. 11: Zone Heatmap & Weekend Dwell
│   └── fig12_huber_delta_sensitivity_and_influence.png # Fig. 12: Huber Delta Sensitivity Sweep
├── results/                                  # Raw reproducible experimental outputs
│   ├── benchmark_results.csv                 # Model comparisons & metrics
│   ├── comprehensive_sota_benchmark.csv      # Cross-validation logs
│   ├── factorial_ablation_study.csv          # Feature ablation results
│   ├── sample_X_test.npy                     # Test split features for exact reproduction
│   └── shap_values.npy                       # Model SHAP explainability values
├── scripts/                                  # Autonomous reproduction scripts
│   ├── export_waybills_dataset.py            # Dataset extraction & perturbation engine
│   ├── run_experiments.py                    # End-to-end model training & evaluation
│   ├── run_q1_rigorous_experiments.py        # Comprehensive SOTA benchmark runner
│   ├── generate_ieee_paper1_figures.py       # Matplotlib publication figures generator
│   ├── calibrate_sota_baselines.py           # Baseline hyperparameter calibration
│   └── reconcile_ols_and_ttest.py            # Statistical hypothesis validation (p < 0.001)
├── CITATION.cff                              # Machine-readable academic citation metadata
├── LICENSE                                   # Dual MIT (Code) & CC BY 4.0 (Data) License
├── requirements.txt                          # Pinned reproducible Python dependencies
└── README.md                                 # Reviewer & replication guide
```

---

## 🚀 Step-by-Step Reproduction Guide

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/ubashir43/topography-freight-eta.git
cd topography-freight-eta

# Create isolated Python virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: .\venv\Scripts\activate

# Install exact pinned dependencies
pip install -r requirements.txt
```

### Step 2: Validate Dataset Integrity
Run Python to verify the SHA-256 integrity of the dataset:
```python
import hashlib

def verify_checksum(filepath, expected):
    with open(filepath, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"Computed SHA-256: {digest}")
    assert digest == expected, "Checksum mismatch!"
    print("Dataset verification: PASSED")

verify_checksum(
    "data/freight_waybills_35k_anonymized.csv",
    "a981c2f10b77e8e4a9194ec710ec4d6189bb5fa5f272a81387d890bfca5cc24a"
)
```

### Step 3: Train Models & Reproduce Benchmark Tables
Execute the automated training and cross-validation pipeline:
```bash
python scripts/run_q1_rigorous_experiments.py
```
This script trains all 8 baseline architectures, calculates 5-fold cross-validation metrics, outputs statistical significance tests ($t$-test and Wilcoxon signed-rank test), and saves results to `results/benchmark_results.csv`.

### Step 4: Regenerate Publication Figures
To regenerate Figures 1 through 12 in 300 DPI publication format:
```bash
python scripts/generate_ieee_paper1_figures.py
```
Output PNGs will be generated in `figures/`.

---

## 💻 Hardware & Environment Specifications

The empirical benchmarks reported in the manuscript were executed under the following standard hardware configuration:
- **Processor:** AMD Ryzen 7 / Intel Core i7 (8 cores, 16 threads @ 3.8 GHz)
- **RAM:** 32 GB DDR4
- **Operating System:** Windows 11 Enterprise (build 26100) / Ubuntu 22.04 LTS verified
- **Python Version:** 3.11.8 (x86_64)
- **BLAS/LAPACK:** OpenBLAS 0.3.26

*Note: All algorithms run CPU-only without requiring CUDA acceleration.*

---

## 📖 Citation

If you utilize this benchmark dataset, code, or methodology in your research, please cite our paper:

```bibtex
@article{bashir2026topography,
  title={Topography-Adaptive Road Circuity and Robust Gradient Boosting for Multi-Carrier Freight Transportation ETA Prediction},
  author={Bashir, Umer and Shoukat, Muhammad Usman and Shams, Muhammad Usman},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026},
  note={Under Peer Review}
}
```

---

## 🧑‍⚖️ Reviewer Reproducibility Statement

This repository is designed to strictly comply with academic reproducibility and open-science standards. All proprietary shipper and consignee entities have been anonymized in accordance with international data protection protocols (GDPR Art. 32 / HIPAA Safe Harbor), while spatial coordinates retain precise road-network circuity, elevation topology, and empirical transit time distributions. All reported numbers in Tables I–IV and Figures 1–12 can be verified deterministically from the provided scripts and dataset.
