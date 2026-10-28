import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import xgboost as xgb
import time

np.random.seed(42)
N = 35000

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
terrain_index = np.clip(np.random.beta(a=2.0, b=3.5, size=N) * 0.85 + 0.05, 0.05, 0.85)

kappa_dynamic = np.clip(1.15 + 0.38 * terrain_index + 0.06 * np.log10(d_geo / 50.0) + np.random.normal(0, 0.02, size=N), 1.12, 1.62)
d_road_dtac = d_geo * kappa_dynamic
d_osrm_true = d_road_dtac * np.random.normal(1.0, 0.015, size=N)

carriers_list = ['DHL', 'FedEx', 'UPS', 'TCS']
carrier_codes = np.random.choice([0, 1, 2, 3], size=N, p=[0.30, 0.30, 0.25, 0.15])
services_list = ['Overnight', 'Express', 'Standard', 'Heavy Freight']
service_codes = np.random.choice([0, 1, 2, 3], size=N, p=[0.15, 0.30, 0.40, 0.15])

carrier_latent_speed = {0: 74.0, 1: 71.5, 2: 70.0, 3: 62.0}
service_latent_priority = {0: 1.25, 1: 1.05, 2: 0.85, 3: 0.70}

effective_speed = np.clip(np.array([carrier_latent_speed[c] * service_latent_priority[s] for c, s in zip(carrier_codes, service_codes)]) + np.random.normal(0, 3.5, size=N), 35.0, 95.0)

weight_kg = np.clip(np.random.exponential(scale=14.0, size=N) + 0.5, 0.5, 180.0)
origin_zone = np.random.randint(0, 6, size=N)
dest_zone = np.random.randint(0, 6, size=N)
hub_congestion = np.random.beta(a=2, b=5, size=N) * 3.0
dow = np.random.randint(0, 7, size=N)
weekend_dwell = np.where(dow >= 5, np.random.uniform(2.5, 4.5, size=N), 0.0)

base_driving_time = d_osrm_true / effective_speed
handling_and_dwell = 1.2 + (weight_kg * 0.018) + hub_congestion + weekend_dwell
y_true = base_driving_time + handling_and_dwell

anomaly_mask = np.random.rand(N) < 0.035
y_true[anomaly_mask] += np.random.exponential(scale=22.0, size=np.sum(anomaly_mask))

# Split indices synchronously so features are 100% aligned
train_idx, test_idx = train_test_split(np.arange(N), test_size=0.20, random_state=42)

# Multi-Feature Euclidean OLS (all 9 features, but straight-line Euclidean distance)
X_tr_euc = np.column_stack([d_geo[train_idx], carrier_codes[train_idx], service_codes[train_idx], weight_kg[train_idx], terrain_index[train_idx], hub_congestion[train_idx], origin_zone[train_idx], dest_zone[train_idx], dow[train_idx]])
X_te_euc = np.column_stack([d_geo[test_idx], carrier_codes[test_idx], service_codes[test_idx], weight_kg[test_idx], terrain_index[test_idx], hub_congestion[test_idx], origin_zone[test_idx], dest_zone[test_idx], dow[test_idx]])

y_train = y_true[train_idx]
y_test = y_true[test_idx]

reg_ols = LinearRegression()
reg_ols.fit(X_tr_euc, y_train)
p_ols = reg_ols.predict(X_te_euc)

mae_ols = mean_absolute_error(y_test, p_ols)
rmse_ols = np.sqrt(mean_squared_error(y_test, p_ols))
r2_ols = r2_score(y_test, p_ols)
comp1_ols = np.mean(np.abs(y_test - p_ols) <= 1.0) * 100.0
comp2_ols = np.mean(np.abs(y_test - p_ols) <= 2.0) * 100.0
mape_ols = np.mean(np.abs((y_test - p_ols) / y_test)) * 100.0

print(f"MATCHED Euclidean Linear OLS (All 9 Features):")
print(f"  MAE: {mae_ols:.2f}h | RMSE: {rmse_ols:.2f}h | MAPE: {mape_ols:.2f}% | <=1h: {comp1_ols:.1f}% | <=2h: {comp2_ols:.1f}% | R2: {r2_ols:.4f}")

# Train D-TAC-GBR and XGBoost for statistical paired t-test
X_tr_dtac = np.column_stack([d_road_dtac[train_idx], carrier_codes[train_idx], service_codes[train_idx], weight_kg[train_idx], terrain_index[train_idx], hub_congestion[train_idx], origin_zone[train_idx], dest_zone[train_idx], dow[train_idx]])
X_te_dtac = np.column_stack([d_road_dtac[test_idx], carrier_codes[test_idx], service_codes[test_idx], weight_kg[test_idx], terrain_index[test_idx], hub_congestion[test_idx], origin_zone[test_idx], dest_zone[test_idx], dow[test_idx]])

dtac_gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, min_samples_leaf=15, loss='huber', alpha=0.90, random_state=42)
dtac_gbr.fit(X_tr_dtac, y_train)
p_dtac = dtac_gbr.predict(X_te_dtac)

xgb_mod = xgb.XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, objective='reg:absoluteerror', random_state=42, n_jobs=-1)
xgb_mod.fit(X_tr_dtac, y_train)
p_xgb = xgb_mod.predict(X_te_dtac)

err_dtac = np.abs(y_test - p_dtac)
err_xgb = np.abs(y_test - p_xgb)

t_stat, p_val = stats.ttest_rel(err_dtac, err_xgb)
diff = np.mean(err_dtac) - np.mean(err_xgb)
print(f"\nPAIRED SIGNIFICANCE TEST (D-TAC-GBR vs XGBoost):")
print(f"  D-TAC MAE: {np.mean(err_dtac):.2f}h | XGBoost MAE: {np.mean(err_xgb):.2f}h")
print(f"  Mean Difference: {diff:.3f}h")
print(f"  Paired t-statistic: t = {t_stat:.3f}, p-value = {p_val:.4f}")
print(f"  Statistically significant difference (alpha=0.05)? {'YES' if p_val < 0.05 else 'NO (STATISTICAL PARITY)'}")

# Update comprehensive_sota_benchmark.csv with the correct OLS numbers
csv_path = r"d:\Logistics Research\Paper1\results\comprehensive_sota_benchmark.csv"
df = pd.read_csv(csv_path)
for idx, r in df.iterrows():
    if 'Euclidean Linear OLS' in r['Model']:
        df.at[idx, 'MAE (h)'] = round(mae_ols, 2)
        df.at[idx, 'RMSE (h)'] = round(rmse_ols, 2)
        df.at[idx, 'MAPE (%)'] = round(mape_ols, 2)
        df.at[idx, 'Compliance (<=1h)'] = round(comp1_ols, 1)
        df.at[idx, 'Compliance (<=2h)'] = round(comp2_ols, 1)
        df.at[idx, 'R2 Score'] = round(r2_ols, 4)

df.to_csv(csv_path, index=False)
print("\nUpdated comprehensive_sota_benchmark.csv successfully with matched OLS!")
