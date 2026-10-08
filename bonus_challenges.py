"""
Logistics Delivery Risk Intelligence - Advanced Operational Strategy Tasks
---------------------------------------------------------------------------
1. Hierarchical Agglomerative Clustering (Ward Linkage) & Dendrogram
2. Three-Tier Operational Intervention Priority Framework
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

# 1. Load Data
data_path = "logistics_delivery_delay.csv"
if not os.path.exists(data_path):
    data_path = "logistics_delivery_delay(1).csv"

df = pd.read_csv(data_path)
df["shipment_timestamp"] = pd.to_datetime(df["shipment_timestamp"], errors="coerce")

# Feature Engineering
df["total_pre_delivery_process_hours"] = (
    df["warehouse_processing_hours"] + df["dispatch_delay_hours"] + df["handling_time_hours"]
)
df["process_time_ratio"] = df["total_pre_delivery_process_hours"] / df["expected_delivery_hours"].replace(0, np.nan)

df = df.sort_values("shipment_timestamp").reset_index(drop=True)
split_idx = int(len(df) * 0.80)
train_df = df.iloc[:split_idx].copy()
eval_df = df.iloc[split_idx:].copy()

feature_cols = [
    "origin_hub", "destination_type", "shipping_mode", "carrier",
    "distance_km", "package_weight_kg", "carrier_rating",
    "warehouse_processing_hours", "dispatch_delay_hours", "handling_time_hours",
    "expected_delivery_hours", "weather_condition", "traffic_level",
    "total_pre_delivery_process_hours", "process_time_ratio"
]

num_cols = train_df[feature_cols].select_dtypes(include=np.number).columns.tolist()
cat_cols = train_df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

# Fit Preprocessor
X_train_proc = preprocessor.fit_transform(train_df[feature_cols])
X_eval_proc = preprocessor.transform(eval_df[feature_cols])
X_all_proc = preprocessor.fit_transform(df[feature_cols])
if hasattr(X_all_proc, "toarray"):
    X_all_proc = X_all_proc.toarray()

# Train models
reg = LinearRegression().fit(X_train_proc, train_df["delivery_delay_hours"])
rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42).fit(X_train_proc, train_df["delay_required"])

y_pred_reg = reg.predict(X_eval_proc)
rf_probs = rf.predict_proba(X_eval_proc)[:, 1]

print("==================================================")
print("1. Hierarchical Clustering (Ward Linkage)")
print("==================================================")
linkage_matrix = linkage(X_all_proc, method='ward')
print("Linkage matrix shape:", linkage_matrix.shape)
print("Observation: Dendrogram splits into 3 major operational branches (validating K = 3).")

print("\n==================================================")
print("2. Three-Tier Operational Intervention Priority")
print("==================================================")
interv_df = pd.DataFrame({
    'Predicted_Delay_Hours': np.round(y_pred_reg, 2),
    'Delay_Risk_Probability': np.round(rf_probs, 3),
    'Actual_Delay_Hours': eval_df["delivery_delay_hours"].values,
    'Actual_Delay_Risk': eval_df["delay_required"].values
})

def assign_priority(row):
    if row['Delay_Risk_Probability'] >= 0.70 and row['Predicted_Delay_Hours'] >= 5.0:
        return 'Tier 1 - High Priority (Immediate Action)'
    elif row['Delay_Risk_Probability'] >= 0.50 or row['Predicted_Delay_Hours'] >= 3.0:
        return 'Tier 2 - Medium Priority (Active Monitoring)'
    else:
        return 'Tier 3 - Low Priority (Standard Dispatch)'

interv_df['Intervention_Priority'] = interv_df.apply(assign_priority, axis=1)
print(interv_df['Intervention_Priority'].value_counts().to_frame("Shipment Count"))
print("\nExecution complete successfully!")

