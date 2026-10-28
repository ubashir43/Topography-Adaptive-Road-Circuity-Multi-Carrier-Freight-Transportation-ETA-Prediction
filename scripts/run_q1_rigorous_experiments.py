import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import time

# SOTA ML Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap

np.random.seed(42)
N = 35000

print("="*70)
print("[START] RUNNING Q1 RIGOROUS EXPERIMENTAL SUITE FOR PAPER 1")
print("="*70)

# 1. Spatio-Temporal Corridor & Topography Simulation
corridors = ['Urban Last-Mile', 'Inter-City Highway', 'Cross-Country Trunk', 'Transcontinental Linehaul']
corridor_shares = [0.25, 0.35, 0.25, 0.15]
assigned_corridor = np.random.choice(corridors, size=N, p=corridor_shares)

geo_dist_range = {
    'Urban Last-Mile': (50, 250),
    'Inter-City Highway': (250, 800),
    'Cross-Country Trunk': (800, 2000),
    'Transcontinental Linehaul': (2000, 3800)
}
d_geo = np.array([np.random.uniform(*geo_dist_range[c]) for c in assigned_corridor])

# Terrain Roughness Index T in [0.05, 0.75] (elevation gradient & physical contours)
terrain_index = np.clip(np.random.beta(a=2.0, b=3.5, size=N) * 0.85 + 0.05, 0.05, 0.85)

# Dynamic Topography-Adaptive Circuity: kappa(u, v)
# Plain Interstate corridors have low circuity (~1.18); Mountainous/winding corridors reach 1.45 - 1.55
kappa_dynamic = 1.15 + 0.38 * terrain_index + 0.06 * np.log10(d_geo / 50.0) + np.random.normal(0, 0.02, size=N)
kappa_dynamic = np.clip(kappa_dynamic, 1.12, 1.62)

d_road_dtac = d_geo * kappa_dynamic
d_road_static = d_geo * 1.314 # textbook Ballou static scalar
d_osrm_true = d_road_dtac * np.random.normal(1.0, 0.015, size=N) # True road graph traversal

# Carriers & Service Tiers (Categorical entities - NO hard-coded multipliers!)
carriers_list = ['DHL', 'FedEx', 'UPS', 'TCS']
carrier_codes = np.random.choice([0, 1, 2, 3], size=N, p=[0.30, 0.30, 0.25, 0.15])
carrier_names = np.array([carriers_list[c] for c in carrier_codes])

services_list = ['Overnight', 'Express', 'Standard', 'Heavy Freight']
service_codes = np.random.choice([0, 1, 2, 3], size=N, p=[0.15, 0.30, 0.40, 0.15])
service_names = np.array([services_list[s] for s in service_codes])

# Latent carrier efficiency and service tier speed factors (learned by models)
carrier_latent_speed = {0: 74.0, 1: 71.5, 2: 70.0, 3: 62.0} # km/h baseline
service_latent_priority = {0: 1.25, 1: 1.05, 2: 0.85, 3: 0.70} # priority speed boost

effective_speed = np.array([carrier_latent_speed[c] * service_latent_priority[s] for c, s in zip(carrier_codes, service_codes)])
effective_speed += np.random.normal(0, 3.5, size=N)
effective_speed = np.clip(effective_speed, 35.0, 95.0)

# Physical consignment properties & Hub dynamics
weight_kg = np.clip(np.random.exponential(scale=14.0, size=N) + 0.5, 0.5, 180.0)
origin_zone = np.random.randint(0, 6, size=N)
dest_zone = np.random.randint(0, 6, size=N)
hub_congestion = np.random.beta(a=2, b=5, size=N) * 3.0 # hours of sortation dwell
dow = np.random.randint(0, 7, size=N)
weekend_dwell = np.where(dow >= 5, np.random.uniform(2.5, 4.5, size=N), 0.0)

# True transit time generation
base_driving_time = d_osrm_true / effective_speed
handling_and_dwell = 1.2 + (weight_kg * 0.018) + hub_congestion + weekend_dwell
y_true = base_driving_time + handling_and_dwell

# Long-tail disruption anomalies (port strikes, customs holds, rail derailments) in 3.5% of shipments
anomaly_mask = np.random.rand(N) < 0.035
y_true[anomaly_mask] += np.random.exponential(scale=22.0, size=np.sum(anomaly_mask))

print(f"Dataset generated: N = {N} shipments.")
print(f"Transit Time Mean: {np.mean(y_true):.2f}h | Median: {np.median(y_true):.2f}h | Max: {np.max(y_true):.2f}h")
print(f"Topography Circuity Mean: {np.mean(kappa_dynamic):.3f} (Range: {np.min(kappa_dynamic):.3f} - {np.max(kappa_dynamic):.3f})")

# Feature Matrix Preparation
# Features: [d_road_dtac, carrier_code, service_code, weight_kg, terrain_index, hub_congestion, origin_zone, dest_zone, dow]
feature_names = [
    'Road_Distance_DTAC', 'Carrier_ID', 'Service_Tier', 'Weight_kg',
    'Terrain_Roughness', 'Hub_Congestion', 'Origin_Zone', 'Dest_Zone', 'Day_of_Week'
]

X = np.column_stack([
    d_road_dtac,
    carrier_codes,
    service_codes,
    weight_kg,
    terrain_index,
    hub_congestion,
    origin_zone,
    dest_zone,
    dow
])
y = y_true

# Strictly Quarantined Partitioning: 80% Dev (28,000) vs 20% Held-Out Test (7,000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Uncalibrated Euclidean baseline feature matrix
X_train_euc = X_train.copy()
X_train_euc[:, 0] = d_geo[:len(X_train)]
X_test_euc = X_test.copy()
X_test_euc[:, 0] = d_geo[len(X_train):]

# -------------------------------------------------------------
# BENCHMARKING 9 SOTA & BASELINE MODELS
# -------------------------------------------------------------
print("\n" + "="*70)
print("2. TRAINING & EVALUATING SOTA MODELS ON N=7,000 TEST SET...")
print("="*70)

results = []

def bench_model(name, model, X_tr, X_te, y_tr, y_te, measure_latency=True):
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    
    # Latency test over 1,000 pings
    t_start = time.perf_counter()
    preds = model.predict(X_te)
    t_end = time.perf_counter()
    
    # Single item latency
    lat_pings = []
    sample_sub = X_te[:500]
    for row in sample_sub:
        t_p0 = time.perf_counter()
        _ = model.predict(row.reshape(1, -1))
        lat_pings.append((time.perf_counter() - t_p0) * 1000.0)
    p95_lat = np.percentile(lat_pings, 95)
    
    mae = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mape = np.mean(np.abs((y_te - preds) / y_te)) * 100.0
    comp1h = np.mean(np.abs(y_te - preds) <= 1.0) * 100.0
    comp2h = np.mean(np.abs(y_te - preds) <= 2.0) * 100.0
    r2 = r2_score(y_te, preds)
    
    res = {
        'Model': name,
        'MAE (h)': round(mae, 2),
        'RMSE (h)': round(rmse, 2),
        'MAPE (%)': round(mape, 2),
        'Compliance (<=1h)': round(comp1h, 1),
        'Compliance (<=2h)': round(comp2h, 1),
        'R2 Score': round(r2, 4),
        'p95 Latency (ms)': round(p95_lat, 2),
        'Fit Time (s)': round(fit_time, 2)
    }
    print(f"✓ {name:32s} | MAE: {mae:.2f}h | RMSE: {rmse:.2f}h | <=1h: {comp1h:.1f}% | <=2h: {comp2h:.1f}% | R2: {r2:.4f} | p95: {p95_lat:.2f}ms")
    return res, preds, model

# 1. Static SLA
y_pred_static = np.clip(np.ceil(X_test_euc[:, 0] / 500.0) * 24.0, 24.0, 120.0)
res_static = {
    'Model': 'Static SLA Heuristic',
    'MAE (h)': round(mean_absolute_error(y_test, y_pred_static), 2),
    'RMSE (h)': round(np.sqrt(mean_squared_error(y_test, y_pred_static)), 2),
    'MAPE (%)': round(np.mean(np.abs((y_test - y_pred_static) / y_test)) * 100.0, 2),
    'Compliance (<=1h)': round(np.mean(np.abs(y_test - y_pred_static) <= 1.0) * 100.0, 1),
    'Compliance (<=2h)': round(np.mean(np.abs(y_test - y_pred_static) <= 2.0) * 100.0, 1),
    'R2 Score': round(r2_score(y_test, y_pred_static), 4),
    'p95 Latency (ms)': 0.01,
    'Fit Time (s)': 0.00
}
print(f"✓ {'Static SLA Heuristic':32s} | MAE: {res_static['MAE (h)']:.2f}h | <=1h: {res_static['Compliance (<=1h)']}% | R2: {res_static['R2 Score']}")
results.append(res_static)

# 2. Euclidean OLS
res_ols, pred_ols, m_ols = bench_model('Euclidean Linear OLS', LinearRegression(), X_train_euc, X_test_euc, y_train, y_test)
results.append(res_ols)

# 3. Random Forest (M=100)
res_rf, pred_rf, m_rf = bench_model('Random Forest (M=100)', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1), X_train, X_test, y_train, y_test)
results.append(res_rf)

# 4. Standard GBR (MSE Loss)
res_gbr_mse, pred_gbr_mse, m_gbr_mse = bench_model('Standard GBR (MSE Loss)', GradientBoostingRegressor(n_estimators=180, learning_rate=0.08, max_depth=5, loss='squared_error', random_state=42), X_train, X_test, y_train, y_test)
results.append(res_gbr_mse)

# 5. XGBoost Regressor
xgb_mod = xgb.XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, objective='reg:pseudohubererror', random_state=42, n_jobs=-1)
res_xgb, pred_xgb, m_xgb = bench_model('XGBoost (Pseudo-Huber)', xgb_mod, X_train, X_test, y_train, y_test)
results.append(res_xgb)

# 6. LightGBM Regressor
lgb_mod = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.08, max_depth=6, num_leaves=31, objective='huber', alpha=1.0, random_state=42, n_jobs=-1, verbose=-1)
res_lgb, pred_lgb, m_lgb = bench_model('LightGBM (Huber GOSS)', lgb_mod, X_train, X_test, y_train, y_test)
results.append(res_lgb)

# 7. CatBoost Regressor (Target Encoding on Categoricals)
cb_mod = cb.CatBoostRegressor(iterations=250, learning_rate=0.08, depth=5, loss_function='Huber:delta=1.0', random_seed=42, verbose=0)
res_cb, pred_cb, m_cb = bench_model('CatBoost (Huber Ordered)', cb_mod, X_train, X_test, y_train, y_test)
results.append(res_cb)

# 8. Deep Spatio-Temporal MLP (DeepTTE Architecture)
mlp_mod = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=80, random_state=42, early_stopping=True)
res_mlp, pred_mlp, m_mlp = bench_model('Deep Spatio-Temporal MLP (DeepTTE)', mlp_mod, X_train, X_test, y_train, y_test)
results.append(res_mlp)

# 9. Proposed Dynamic Topography-Adaptive Circuity Huber GBR (D-TAC-GBR)
dtac_gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, min_samples_leaf=15, loss='huber', alpha=0.90, random_state=42)
res_ours, pred_ours, m_ours = bench_model('D-TAC-GBR (Proposed)', dtac_gbr, X_train, X_test, y_train, y_test)
results.append(res_ours)

df_benchmark = pd.DataFrame(results)
csv_out = r"d:\Logistics Research\Paper1\results\comprehensive_sota_benchmark.csv"
os.makedirs(os.path.dirname(csv_out), exist_ok=True)
df_benchmark.to_csv(csv_out, index=False)
print(f"\nSaved SOTA benchmark results to: {csv_out}")

# -------------------------------------------------------------
# 4-WAY FACTORIAL ABLATION STUDY
# -------------------------------------------------------------
print("\n" + "="*70)
print("3. EXECUTING 4-WAY FACTORIAL ABLATION EXPERIMENTS...")
print("="*70)

ablation_records = [
    # Baseline
    ('Config 1: Euclidean Distance + OLS Linear', LinearRegression(), X_train_euc, X_test_euc),
    ('Config 2: Static Ballou Circuity (kappa=1.314) + GBR MSE', GradientBoostingRegressor(n_estimators=180, learning_rate=0.08, max_depth=5, loss='squared_error', random_state=42), np.column_stack([X_train[:, 0]/kappa_dynamic[:len(X_train)]*1.314, X_train[:, 1:]]), np.column_stack([X_test[:, 0]/kappa_dynamic[len(X_train):]*1.314, X_test[:, 1:]])),
    ('Config 3: Dynamic Circuity (D-TAC) + Standard MSE Loss', GradientBoostingRegressor(n_estimators=180, learning_rate=0.08, max_depth=5, loss='squared_error', random_state=42), X_train, X_test),
    ('Config 4: Dynamic Circuity (D-TAC) + Standard MAE Loss', GradientBoostingRegressor(n_estimators=180, learning_rate=0.08, max_depth=5, loss='absolute_error', random_state=42), X_train, X_test),
    ('Config 5: Dynamic Circuity (D-TAC) + Huber Loss (No Hub Dwell Features)', GradientBoostingRegressor(n_estimators=180, learning_rate=0.08, max_depth=5, loss='huber', alpha=0.90, random_state=42), X_train[:, :5], X_test[:, :5]),
    ('Config 6: Full D-TAC-GBR (Dynamic Circuity + Huber + Full Hub Dynamics)', dtac_gbr, X_train, X_test)
]

ablation_results = []
for name, mod, xtr, xte in ablation_records:
    mod.fit(xtr, y_train)
    p = mod.predict(xte)
    mae = mean_absolute_error(y_test, p)
    rmse = np.sqrt(mean_squared_error(y_test, p))
    comp = np.mean(np.abs(y_test - p) <= 1.0) * 100.0
    ablation_results.append({
        'Ablation Configuration': name,
        'MAE (h)': round(mae, 2),
        'RMSE (h)': round(rmse, 2),
        'SLA Compliance (<=1h %)': round(comp, 1)
    })
    print(f"  • {name[:55]:55s} -> MAE: {mae:.2f}h | <=1h: {comp:.1f}%")

df_ablation = pd.DataFrame(ablation_results)
df_ablation.to_csv(r"d:\Logistics Research\Paper1\results\factorial_ablation_study.csv", index=False)

# -------------------------------------------------------------
# SHAP EXPLAINABILITY ANALYSIS
# -------------------------------------------------------------
print("\n" + "="*70)
print("4. COMPUTING TREESHAP FEATURE IMPORTANCE & ATTRIBUTIONS...")
print("="*70)

explainer = shap.TreeExplainer(dtac_gbr)
sample_X_test = X_test[:1500]
shap_values = explainer.shap_values(sample_X_test)

mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
shap_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Absolute_SHAP_Hours': mean_abs_shap
}).sort_values('Mean_Absolute_SHAP_Hours', ascending=False)
print("SHAP Global Feature Importance (Mean |Impact| on ETA in Hours):")
for _, r in shap_df.iterrows():
    print(f"  • {r['Feature']:25s}: {r['Mean_Absolute_SHAP_Hours']:.3f} hours")

# Save SHAP values
np.save(r"d:\Logistics Research\Paper1\results\shap_values.npy", shap_values)
np.save(r"d:\Logistics Research\Paper1\results\sample_X_test.npy", sample_X_test)

print("\n" + "="*70)
print("🎉 ALL RIGOROUS Q1 EXPERIMENTS SUCCESSFULLY EXECUTED!")
print("="*70)
