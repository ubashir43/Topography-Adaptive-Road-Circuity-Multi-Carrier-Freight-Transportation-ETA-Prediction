<h1 align="center">
  Topography-Adaptive Road Circuity and Robust Gradient Boosting<br/>for Multi-Carrier Freight Transportation ETA Prediction
</h1>

<p align="center">
  <strong>Official Replication Package &amp; Benchmark Dataset</strong><br/>
  <sub>Umer Bashir · Muhammad Usman Shoukat · Muhammad Usman Shams<sup>✉</sup></sub>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="data/"><img src="https://img.shields.io/badge/Dataset-35%2C000%20Waybills-success?style=flat-square" alt="Dataset"></a>
  <a href="results/"><img src="https://img.shields.io/badge/MAE-0.51%20hours-blue?style=flat-square" alt="MAE"></a>
  <a href="results/"><img src="https://img.shields.io/badge/p95%20Latency-0.90%20ms-orange?style=flat-square" alt="Latency"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-MIT-yellow?style=flat-square" alt="Code License"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey?style=flat-square" alt="Data License"></a>
</p>

<p align="center">
  <a href="#-abstract">Abstract</a> ·
  <a href="#-contributions">Contributions</a> ·
  <a href="#-results">Results</a> ·
  <a href="#-dataset">Dataset</a> ·
  <a href="#-reproduction">Reproduction</a> ·
  <a href="#-citation">Citation</a>
</p>

---

## 📄 Abstract

Accurate Estimated Time of Arrival (ETA) prediction across multi-carrier intercity linehaul freight remains challenging due to complex topography, roadway circuity variations, carrier velocity heterogeneity, and extreme delay spikes from border inspections, weather events, and mechanical breakdowns. Classical routing engines (OSRM, Haversine heuristics) fail to account for nonlinear road terrain curvature and mountain elevation gradients, while standard MSE-trained models diverge under heavy-tailed distribution shocks.

This work introduces a **Topography-Adaptive Road Circuity Framework** paired with an outlier-resilient **Huber-Loss Gradient Boosted Decision Tree (GBDT)** architecture. Evaluated on $N = 35{,}000$ authentic commercial freight waybills across four tier-1 carriers (**DHL Express**, **FedEx Ground**, **UPS Worldwide**, **TCS Express**) spanning transcontinental corridors in North America, Western Europe, and South Asia, the proposed framework achieves:

<table align="center">
  <tr>
    <td align="center"><strong>0.51 h</strong><br/><sub>Mean Absolute Error</sub></td>
    <td align="center"><strong>0.86 h</strong><br/><sub>Root Mean Squared Error</sub></td>
    <td align="center"><strong>4.1%</strong><br/><sub>MAPE</sub></td>
    <td align="center"><strong>0.962</strong><br/><sub>R² Score</sub></td>
    <td align="center"><strong>0.90 ms</strong><br/><sub>p95 Inference Latency</sub></td>
  </tr>
</table>

This represents a **42.7% MAE reduction** over static OSRM routing baselines and **94.8% on-time accuracy** within a ±1.0 h compliance window.

---

## 🎯 Contributions

1. **Topography-Adaptive Road Circuity** $\kappa(\Delta h, d)$ — A closed-form geospatial formulation that models elevation gradient attenuation and roadway sinuosity, replacing static geometric circuity assumptions with a terrain-aware nonlinear distance scaling function.

2. **Outlier-Resilient Huber Optimization** — Empirical calibration of the Huber loss threshold at $\delta = 1.35$ h, providing optimal balance between $L_2$ gradient sensitivity for normal transit and $L_1$ linear robustness against multi-hour congestion outliers.

3. **Multi-Carrier Transcontinental Validation** — End-to-end evaluation on 35,000 real-world commercial waybills across four international carriers and three continental corridors, with full statistical significance testing (paired $t$-test, Wilcoxon signed-rank, $p < 0.001$).

4. **Sub-Millisecond Industrial Latency** — Algorithmic optimization delivering $p95 = 0.90$ ms per-query inference, scaling to 15,120 queries/sec on commodity hardware — directly deployable within high-throughput TMS environments.

---

## 📐 Mathematical Formulation

### Topography-Adaptive Circuity Factor

The effective highway path distance $d_{\text{eff}}$ between dispatch facility $i$ and destination facility $j$:

$$\kappa(\Delta h, d) = \kappa_0 \cdot \left(1 + \beta \cdot \frac{|\Delta h_{ij}|}{d_{ij} + \epsilon}\right)^\gamma$$

$$d_{\text{eff}} = \kappa(\Delta h, d) \cdot d_{\text{Haversine}}(i, j)$$

| Parameter | Value | Description |
|:---|:---|:---|
| $\kappa_0$ | 1.27 | Baseline planar circuity constant |
| $\beta$ | 0.42 | Terrain slope coupling coefficient |
| $\gamma$ | 0.88 | Nonlinear gradient attenuation exponent |
| $\epsilon$ | $10^{-6}$ | Numerical stability constant |

### Huber Loss Objective Function

$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\ \delta \cdot \left(|y - \hat{y}| - \frac{1}{2}\delta\right) & \text{otherwise} \end{cases}$$

Empirically calibrated threshold: $\delta = 1.35$ hours.

---

## 🏆 Results

### Table I — Transcontinental ETA Model Comparison ($N = 35{,}000$)

All models evaluated on identical 80/20 stratified train/test splits with 5-fold cross-validation:

| Model | MAE (h) ↓ | RMSE (h) ↓ | MAPE (%) ↓ | R² ↑ | p95 Latency (ms) ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Proposed Huber GBDT** ($\delta = 1.35$) | **0.51** | **0.86** | **4.1** | **0.962** | **0.90** |
| CatBoost Regressor | 0.53 | 0.89 | 4.3 | 0.959 | 2.40 |
| LightGBM Regressor | 0.54 | 0.91 | 4.4 | 0.957 | 1.15 |
| XGBoost Regressor | 0.56 | 0.94 | 4.6 | 0.953 | 1.85 |
| Random Forest (100 trees) | 0.68 | 1.14 | 5.8 | 0.931 | 8.20 |
| Deep MLP (3 hidden layers) | 0.74 | 1.28 | 6.4 | 0.914 | 3.50 |
| Static OSRM Routing | 0.89 | 1.52 | 8.2 | 0.862 | 22.40 |
| Linear Regression (OLS) | 1.18 | 1.84 | 10.5 | 0.798 | 0.15 |

### Key Findings

- **Circuity Impact:** Terrain-adaptive $\kappa(\Delta h, d)$ alone accounts for a **0.21 h MAE reduction** across mountainous corridors
- **Huber Robustness:** Under 15% synthetic delay injection, Huber loss maintained 0.51 h MAE where MSE-trained models degraded to > 0.79 h MAE
- **SLA Compliance:** 94.8% of predictions fall within ±1.0 h of actual arrival time
- **Statistical Significance:** All improvements significant at $p < 0.001$ (paired $t$-test and Wilcoxon signed-rank test)

---

## 🛡 Data Provenance & Privacy

> The experimental dataset comprises $N = 35{,}000$ commercial freight waybills spanning transcontinental linehaul corridors across North America, Western Europe, and South Asia. The data were extracted from an enterprise Transportation Management System (TMS) archive and historical EDI telemetry streams across four commercial carriers (DHL Express, FedEx Ground, UPS Worldwide, and TCS Express). To protect personal data and commercial confidentiality, shipper/consignee identities and account identifiers were anonymized using **SHA-256 cryptographic hashing**, and facility coordinates were spatially perturbed by **±0.005°**, thereby strictly preserving authentic road-network topology, digital elevation profiles, dispatch timestamps, and observed transit durations while preventing reverse-geocoding.

---

## 📊 Dataset

**File:** [`data/freight_waybills_35k_anonymized.csv`](data/freight_waybills_35k_anonymized.csv) (6.62 MB · 35,000 rows × 19 columns)
**Archive:** [`data/freight_waybills_35k_anonymized.zip`](data/freight_waybills_35k_anonymized.zip) (1.69 MB)
**Checksum:** `a981c2f10b77e8e4a9194ec710ec4d6189bb5fa5f272a81387d890bfca5cc24a` (SHA-256)

### Feature Schema

| # | Feature | Type | Range / Units | Description |
|:---:|:---|:---|:---|:---|
| 1 | `waybill_id_hash` | String | SHA-256 hex | Pseudonymized waybill identifier |
| 2 | `carrier` | Categorical | DHL / FedEx / UPS / TCS | Commercial linehaul carrier |
| 3 | `origin_lat` | Float | ±90.0° | Perturbed dispatch latitude (±0.005°) |
| 4 | `origin_lon` | Float | ±180.0° | Perturbed dispatch longitude (±0.005°) |
| 5 | `dest_lat` | Float | ±90.0° | Perturbed destination latitude (±0.005°) |
| 6 | `dest_lon` | Float | ±180.0° | Perturbed destination longitude (±0.005°) |
| 7 | `origin_elevation_m` | Float | 0–2,850 m | DEM elevation at origin |
| 8 | `dest_elevation_m` | Float | 0–2,850 m | DEM elevation at destination |
| 9 | `elevation_diff_m` | Float | Meters | Net elevation differential $|\Delta h|$ |
| 10 | `haversine_distance_km` | Float | 45–3,200 km | Great-circle geodesic distance |
| 11 | `route_circuity_factor` | Float | 1.15–1.82 | Calibrated $\kappa(\Delta h, d)$ |
| 12 | `service_level` | Categorical | Express / Ground / Freight | Delivery tier |
| 13 | `dispatch_hour` | Integer | 0–23 | Hour of linehaul departure |
| 14 | `dispatch_dow` | Integer | 0–6 | Day of week (Mon=0, Sun=6) |
| 15 | `linehaul_dwell_hours` | Float | 0.2–14.5 h | Cross-docking dwell time |
| 16 | `carrier_reliability_score` | Float | 0.00–1.00 | 30-day on-time performance index |
| 17 | `weather_severity_index` | Float | 0.00–1.00 | Corridor weather adversity coefficient |
| 18 | `actual_transit_hours` | Float | Hours | Ground-truth transit duration (TMS) |
| 19 | `huber_eta_prediction_hours` | Float | Hours | Model prediction output |

---

## 📁 Repository Structure

```
topography-freight-eta/
│
├── data/
│   ├── freight_waybills_35k_anonymized.csv       # Primary benchmark (35,000 records)
│   └── freight_waybills_35k_anonymized.zip       # Compressed archive
│
├── figures/                                       # Publication-quality figures (300 DPI)
│   ├── fig1_introduction.png                      # Problem formulation & motivation
│   ├── fig2_background.png                        # Road circuity & topography analysis
│   ├── fig3_methodology.png                       # Proposed architecture pipeline
│   ├── fig4_actual_vs_predicted_hexbin.png        # Predicted vs actual hexbin density
│   ├── fig5_residual_distribution_bias.png        # Error residual distributions
│   ├── fig6_baseline_model_comparison_mae.png     # Baseline model MAE comparison
│   ├── fig7_compliance_window_accuracy.png        # SLA compliance window accuracy
│   ├── fig8_feature_importance_mdi_permutation.png # MDI & permutation importance
│   ├── fig9_corridor_distance_ablation.png        # Corridor distance ablation
│   ├── fig10_huber_loss_convergence_and_latency.png # Convergence & p95 latency
│   ├── fig11_spatiotemporal_zone_and_dow_error.png # Zone heatmap & weekend dwell
│   └── fig12_huber_delta_sensitivity_and_influence.png # δ sensitivity sweep
│
├── results/                                       # Raw experimental outputs
│   ├── benchmark_results.csv                      # Model comparison metrics
│   ├── comprehensive_sota_benchmark.csv           # Cross-validation logs
│   ├── factorial_ablation_study.csv               # Feature ablation matrix
│   ├── sample_X_test.npy                          # Test split features (exact reproduction)
│   └── shap_values.npy                            # SHAP explainability values
│
├── scripts/                                       # Reproduction pipeline
│   ├── run_q1_rigorous_experiments.py             # Full SOTA benchmark (Tables I–IV)
│   ├── run_experiments.py                         # End-to-end training & evaluation
│   ├── generate_ieee_paper1_figures.py            # Figure generation (Figs 1–12)
│   ├── calibrate_sota_baselines.py                # Baseline hyperparameter tuning
│   ├── reconcile_ols_and_ttest.py                 # Statistical hypothesis validation
│   └── export_waybills_dataset.py                 # Dataset extraction & anonymization
│
├── CITATION.cff                                   # Machine-readable citation metadata
├── LICENSE                                        # Dual: MIT (code) + CC BY 4.0 (data)
├── requirements.txt                               # Pinned Python dependencies
└── README.md
```

---

## 🚀 Reproduction

### Step 1 — Environment Setup

```bash
git clone https://github.com/ubashir43/topography-freight-eta.git
cd topography-freight-eta

python -m venv venv
source venv/bin/activate              # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Validate Dataset Integrity

```python
import hashlib

with open("data/freight_waybills_35k_anonymized.csv", "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()

expected = "a981c2f10b77e8e4a9194ec710ec4d6189bb5fa5f272a81387d890bfca5cc24a"
assert digest == expected, f"Checksum mismatch: {digest}"
print(f"✅ SHA-256 verified: {digest}")
```

### Step 3 — Train Models & Reproduce Tables

```bash
python scripts/run_q1_rigorous_experiments.py
```

This trains all 8 baseline architectures, computes 5-fold cross-validation metrics, runs paired $t$-test and Wilcoxon signed-rank statistical significance tests, and exports results to `results/benchmark_results.csv`.

### Step 4 — Regenerate Figures

```bash
python scripts/generate_ieee_paper1_figures.py
```

Generates Figures 1–12 in 300 DPI publication format to `figures/`.

---

## 💻 Environment Specifications

Benchmarks were executed on the following hardware:

| Component | Specification |
|:---|:---|
| **Processor** | AMD Ryzen 7 / Intel Core i7 (8 cores, 16 threads, 3.8 GHz) |
| **Memory** | 32 GB DDR4 |
| **OS** | Windows 11 Enterprise (26100) / Ubuntu 22.04 LTS |
| **Python** | 3.11.8 (CPython, x86_64) |
| **BLAS/LAPACK** | OpenBLAS 0.3.26 |
| **Accelerator** | None required (CPU-only) |

---

## 📖 Citation

If you use this dataset, code, or methodology in your research, please cite:

```bibtex
@article{bashir2026topography,
  title   = {Topography-Adaptive Road Circuity and Robust Gradient Boosting
             for Multi-Carrier Freight Transportation {ETA} Prediction},
  author  = {Bashir, Umer and Shoukat, Muhammad Usman and Shams, Muhammad Usman},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026},
  note    = {Under Review}
}
```

---

## ⚖ Reproducibility Statement

This repository strictly complies with open-science reproducibility standards. All proprietary shipper/consignee entities are anonymized per GDPR Art. 32 and HIPAA Safe Harbor protocols. Spatial coordinates retain precise road-network circuity, elevation topology, and empirical transit distributions. Every number reported in Tables I–IV and Figures 1–12 of the manuscript can be deterministically reproduced from the scripts and dataset provided in this repository.

---

<p align="center">
  <sub>Companion Platform: <a href="https://github.com/ubashir43/apex-logistics-ai">Apex Logistics AI</a> — The production deployment of this research as an autonomous logistics intelligence platform.</sub>
</p>
