import pandas as pd
import numpy as np
from xgboost import XGBRegressor   # ✅ CHANGED
import joblib
import warnings

warnings.filterwarnings("ignore")

print("--- Starting Final M2 (XGBoost, Monthly) Training ---")

MODEL_NAME = "A2\\M2_monthly.joblib"
DATA_SOURCE_RAW = "A2\\A2.csv"
DATA_SOURCE_PROCESSED = "A2\\A2_monthly_calls.csv"

# --- 0. Feature Creation Function (for MONTHLY data) ---
def create_features(df, target_col):
    """Create time-based and lag features for MONTHLY data."""
    df_feat = df.copy()
    
    df_feat["month"] = df_feat.index.month
    df_feat["quarter"] = df_feat.index.quarter
    df_feat["year"] = df_feat.index.year

    # Lag feature based on quarterly seasonality (3 months)
    df_feat["lag_3"] = df_feat[target_col].shift(3)
    
    # Rolling mean of previous 3 months
    df_feat["rolling_mean_3"] = (
        df_feat[target_col].shift(1).rolling(window=3).mean()
    )

    df_feat = df_feat.dropna()
    return df_feat.drop(target_col, axis=1), df_feat[target_col]

# --- 1. Preprocessing ---
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")
try:
    df_raw = pd.read_csv(DATA_SOURCE_RAW)

    df_raw["datetime"] = pd.to_datetime(
        df_raw["Time of Call"],
        format="mixed",
        dayfirst=False
    )
    df_raw = df_raw.set_index("datetime")
    
    # Monthly aggregation
    df_monthly = df_raw.resample("M").size().to_frame("call_count")
    df_monthly = df_monthly.asfreq("M", fill_value=0)
    
    # Drop the last incomplete month
    if not df_monthly.empty:
        df_monthly = df_monthly.iloc[:-1]
    
    df_monthly.to_csv(DATA_SOURCE_PROCESSED)
    print(f"Preprocessed and saved '{DATA_SOURCE_PROCESSED}'. Total {len(df_monthly)} months.")
    
except Exception as e:
    print(f"Error during preprocessing: {e}")
    raise

# --- 2. Feature Engineering ---
print("Creating features from 100% of the monthly data...")
X, y = create_features(df_monthly, "call_count")

print(f"Training features shape: {X.shape}")
print(f"Training target shape: {y.shape}")

if X.empty:
    print("FATAL: No features were created. Data shorter than 3 months.")
else:
    # --- 3. Model Training ---
    print("Training XGBoost model on full monthly dataset...")
    
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X, y)
    print("Model training complete.")

    # --- 4. Save Model ---
    print(f"Saving model to '{MODEL_NAME}'...")
    joblib.dump(model, MODEL_NAME)
    print(f"Successfully saved '{MODEL_NAME}'.")

print("--- Monthly Training Script Finished ---")
