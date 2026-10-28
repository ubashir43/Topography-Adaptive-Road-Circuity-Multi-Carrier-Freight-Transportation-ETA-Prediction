import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

np.random.seed(42)
N = 35000

# Same dataset generation
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
kappa_dynamic = 1.15 + 0.38 * terrain_index + 0.06 * np.log10(d_geo / 50.0) + np.random.normal(0, 0.02, size=N)
kappa_dynamic = np.clip(kappa_dynamic, 1.12, 1.62)

d_road_dtac = d_geo * kappa_dynamic
d_osrm_true = d_road_dtac * np.random.normal(1.0, 0.015, size=N)

carriers_list = ['DHL', 'FedEx', 'UPS', 'TCS']
carrier_codes = np.random.choice([0, 1, 2, 3], size=N, p=[0.30, 0.30, 0.25, 0.15])
services_list = ['Overnight', 'Express', 'Standard', 'Heavy Freight']
service_codes = np.random.choice([0, 1, 2, 3], size=N, p=[0.15, 0.30, 0.40, 0.15])

carrier_latent_speed = {0: 74.0, 1: 71.5, 2: 70.0, 3: 62.0}
service_latent_priority = {0: 1.25, 1: 1.05, 2: 0.85, 3: 0.70}

effective_speed = np.array([carrier_latent_speed[c] * service_latent_priority[s] for c, s in zip(carrier_codes, service_codes)])
effective_speed += np.random.normal(0, 3.5, size=N)
effective_speed = np.clip(effective_speed, 35.0, 95.0)

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

X = np.column_stack([
    d_road_dtac, carrier_codes, service_codes, weight_kg,
    terrain_index, hub_congestion, origin_zone, dest_zone, dow
])
y = y_true

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

def eval_mod(name, model):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    
    preds = model.predict(X_test)
    
    lat_pings = []
    for row in X_test[:500]:
        t_p0 = time.perf_counter()
        _ = model.predict(row.reshape(1, -1))
        lat_pings.append((time.perf_counter() - t_p0) * 1000.0)
    p95_lat = np.percentile(lat_pings, 95)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100.0
    comp1h = np.mean(np.abs(y_test - preds) <= 1.0) * 100.0
    comp2h = np.mean(np.abs(y_test - preds) <= 2.0) * 100.0
    r2 = r2_score(y_test, preds)
    
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
    print(f"Result for {name}: MAE={mae:.2f}h, RMSE={rmse:.2f}h, <=1h={comp1h:.1f}%, <=2h={comp2h:.1f}%, p95={p95_lat:.2f}ms")
    return res

# 1. XGBoost with absolute error (robust to long-tail)
xgb_mod = xgb.XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, objective='reg:absoluteerror', random_state=42, n_jobs=-1)
res_xgb = eval_mod('XGBoost (Robust L1)', xgb_mod)

# 2. LightGBM with regression_l1
lgb_mod = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.08, max_depth=6, num_leaves=31, objective='regression_l1', random_state=42, n_jobs=-1, verbose=-1)
res_lgb = eval_mod('LightGBM (Robust L1)', lgb_mod)

# 3. CatBoost with MAE
cb_mod = cb.CatBoostRegressor(iterations=250, learning_rate=0.08, depth=5, loss_function='MAE', random_seed=42, verbose=0)
res_cb = eval_mod('CatBoost (Ordered MAE)', cb_mod)

# Update CSV
csv_path = r"d:\Logistics Research\Paper1\results\comprehensive_sota_benchmark.csv"
df = pd.read_csv(csv_path)

# Replace XGBoost, LightGBM, CatBoost rows
for idx, r in df.iterrows():
    if 'XGBoost' in r['Model']:
        for k, v in res_xgb.items():
            df.at[idx, k] = v
    elif 'LightGBM' in r['Model']:
        for k, v in res_lgb.items():
            df.at[idx, k] = v
    elif 'CatBoost' in r['Model']:
        for k, v in res_cb.items():
            df.at[idx, k] = v

df.to_csv(csv_path, index=False)
print("Updated comprehensive_sota_benchmark.csv successfully!")
