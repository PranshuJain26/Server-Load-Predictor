import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import warnings

warnings.filterwarnings("ignore")

print("--- Starting Final M2 (XGBoost, Hourly) Training ---")

MODEL_NAME = "A2\\M2_hourly.joblib"
DATA_SOURCE_RAW = "A2\\A2.csv"
DATA_SOURCE_PROCESSED = "A2\\A2_hourly_calls.csv"

# --------------------------------------------------
# 0. Feature Creation Function (HOURLY)
# --------------------------------------------------
def create_features(df, target_col):
    """Create time-based and lag features for HOURLY data."""
    df_feat = df.copy()

    df_feat["hour"] = df_feat.index.hour
    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month

    # 24-hour lag features
    df_feat["lag_24"] = df_feat[target_col].shift(24)
    df_feat["rolling_mean_24"] = (
        df_feat[target_col].shift(1).rolling(window=24).mean()
    )

    # Drop NaNs created by lags
    df_feat = df_feat.dropna()

    return df_feat.drop(target_col, axis=1), df_feat[target_col]

# --------------------------------------------------
# 1. Preprocessing
# --------------------------------------------------
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")

try:
    df_raw = pd.read_csv(DATA_SOURCE_RAW)

    # ✅ FIXED: robust datetime parsing (matches your CSV)
    df_raw["datetime"] = pd.to_datetime(
        df_raw["Time of Call"],
        format="mixed",
        dayfirst=False
    )

    df_raw = df_raw.set_index("datetime")

    # Hourly aggregation (each row = 1 call)
    df_hourly = df_raw.resample("H").size().to_frame("call_count")
    df_hourly = df_hourly.asfreq("H", fill_value=0)

    # Drop last incomplete hour
    if not df_hourly.empty:
        df_hourly = df_hourly.iloc[:-1]

    df_hourly.to_csv(DATA_SOURCE_PROCESSED)
    print(f"Preprocessed and saved '{DATA_SOURCE_PROCESSED}'. Total {len(df_hourly)} hours.")

except Exception as e:
    print(f"Error during preprocessing: {e}")
    raise

# --------------------------------------------------
# 2. Feature Engineering
# --------------------------------------------------
print("Creating features from 100% of the hourly data...")
X, y = create_features(df_hourly, "call_count")

print(f"Training features shape: {X.shape}")
print(f"Training target shape: {y.shape}")

if X.empty:
    print("FATAL: No features created (need at least 24 hours of data).")
else:
    # --------------------------------------------------
    # 3. Model Training (XGBoost)
    # --------------------------------------------------
    print("Training XGBoost model on full hourly dataset...")

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)
    print("Model training complete.")

    # --------------------------------------------------
    # 4. Save Model
    # --------------------------------------------------
    print(f"Saving model to '{MODEL_NAME}'...")
    joblib.dump(model, MODEL_NAME)
    print(f"Successfully saved '{MODEL_NAME}'.")

print("--- Hourly Training Script Finished ---")
