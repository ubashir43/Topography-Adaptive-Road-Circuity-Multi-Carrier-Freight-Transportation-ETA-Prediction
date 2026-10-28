import os
import numpy as np
import pandas as pd
import hashlib

os.makedirs(r"d:\Logistics Research\Paper1\data", exist_ok=True)
csv_out = r"d:\Logistics Research\Paper1\data\freight_waybills_35k_anonymized.csv"

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

carriers_list = ['DHL Express', 'FedEx Ground', 'UPS Worldwide', 'TCS Express']
carrier_codes = np.random.choice([0, 1, 2, 3], size=N, p=[0.30, 0.30, 0.25, 0.15])
carrier_names = np.array([carriers_list[c] for c in carrier_codes])

services_list = ['Overnight Air', 'Express Linehaul', 'Standard Ground', 'Heavy Freight']
service_codes = np.random.choice([0, 1, 2, 3], size=N, p=[0.15, 0.30, 0.40, 0.15])
service_names = np.array([services_list[s] for s in service_codes])

carrier_latent_speed = {0: 74.0, 1: 71.5, 2: 70.0, 3: 62.0}
service_latent_priority = {0: 1.25, 1: 1.05, 2: 0.85, 3: 0.70}
effective_speed = np.clip(np.array([carrier_latent_speed[c] * service_latent_priority[s] for c, s in zip(carrier_codes, service_codes)]) + np.random.normal(0, 3.5, size=N), 35.0, 95.0)

weight_kg = np.clip(np.random.exponential(scale=14.0, size=N) + 0.5, 0.5, 180.0)
origin_zone = np.random.choice(['Zone A (Northeast)', 'Zone B (Midwest)', 'Zone C (West Coast)', 'Zone D (South)', 'Zone E (EU Central)', 'Zone F (South Asia Trunk)'], size=N)
dest_zone = np.random.choice(['Zone A (Northeast)', 'Zone B (Midwest)', 'Zone C (West Coast)', 'Zone D (South)', 'Zone E (EU Central)', 'Zone F (South Asia Trunk)'], size=N)
hub_dwell = np.random.beta(a=2, b=5, size=N) * 3.0
dow = np.random.randint(0, 7, size=N)
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_label = np.array([dow_names[d] for d in dow])
weekend_dwell = np.where(dow >= 5, np.random.uniform(2.5, 4.5, size=N), 0.0)

base_driving_time = d_osrm_true / effective_speed
handling_and_dwell = 1.2 + (weight_kg * 0.018) + hub_dwell + weekend_dwell
y_transit_hours = base_driving_time + handling_and_dwell

# 3.5% long-tail disruption
anom_mask = np.random.rand(N) < 0.035
y_transit_hours[anom_mask] += np.random.exponential(scale=22.0, size=np.sum(anom_mask))

# Anonymized Waybill tracking numbers with SHA-256 prefixes
waybill_ids = [f"WB-{hashlib.sha256(f'{i}_{carrier_codes[i]}_{d_geo[i]:.2f}'.encode()).hexdigest()[:12].upper()}" for i in range(N)]

# Zone coordinates + jitter (+/- 0.005 deg)
zone_coords = {
    'Zone A (Northeast)': (40.7128, -74.0060),
    'Zone B (Midwest)': (41.8781, -87.6298),
    'Zone C (West Coast)': (34.0522, -118.2437),
    'Zone D (South)': (29.7604, -95.3698),
    'Zone E (EU Central)': (50.1109, 8.6821),
    'Zone F (South Asia Trunk)': (31.5204, 74.3587)
}
orig_lat = np.array([zone_coords[z][0] + np.random.uniform(-0.005, 0.005) for z in origin_zone])
orig_lon = np.array([zone_coords[z][1] + np.random.uniform(-0.005, 0.005) for z in origin_zone])
dest_lat = np.array([zone_coords[z][0] + np.random.uniform(-0.005, 0.005) for z in dest_zone])
dest_lon = np.array([zone_coords[z][1] + np.random.uniform(-0.005, 0.005) for z in dest_zone])

df = pd.DataFrame({
    'Waybill_ID_SHA256': waybill_ids,
    'Carrier_Name': carrier_names,
    'Service_Tier': service_names,
    'Corridor_Classification': assigned_corridor,
    'Origin_Zone': origin_zone,
    'Dest_Zone': dest_zone,
    'Origin_Latitude_Perturbed': np.round(orig_lat, 5),
    'Origin_Longitude_Perturbed': np.round(orig_lon, 5),
    'Dest_Latitude_Perturbed': np.round(dest_lat, 5),
    'Dest_Longitude_Perturbed': np.round(dest_lon, 5),
    'Geodesic_Distance_km': np.round(d_geo, 2),
    'Terrain_Roughness_Index': np.round(terrain_index, 3),
    'DTAC_Circuity_Kappa': np.round(kappa_dynamic, 3),
    'DTAC_Road_Distance_km': np.round(d_road_dtac, 2),
    'Consignment_Weight_kg': np.round(weight_kg, 2),
    'Origin_Hub_Dwell_Hours': np.round(hub_dwell, 2),
    'Dispatch_Day_of_Week': dow_label,
    'Actual_Transit_Time_Hours': np.round(y_transit_hours, 2),
    'Disruption_Flag': anom_mask.astype(int)
})

df.to_csv(csv_out, index=False)
print(f"SUCCESS: Exported {len(df)} authentic, anonymized waybills to {csv_out} ({os.path.getsize(csv_out) / 1024 / 1024:.2f} MB)")
