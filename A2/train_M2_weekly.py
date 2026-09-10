import pandas as pd
import numpy as np
from xgboost import XGBRegressor   # ✅ CHANGED
import joblib
import warnings

warnings.filterwarnings("ignore")

print("--- Starting Final M2 (XGBoost, Weekly) Training ---")

MODEL_NAME = "A2\\M2_weekly.joblib"
DATA_SOURCE_RAW = "A2\\A2.csv"
DATA_SOURCE_PROCESSED = "A2\\A2_weekly_calls.csv"

# --- 0. Feature Creation Function (for WEEKLY data) ---
def create_features(df, target_col):
    """Create time-based and lag features for WEEKLY data."""
    df_feat = df.copy()
    
    df_feat["month"] = df_feat.index.month
    df_feat["weekofyear"] = df_feat.index.isocalendar().week.astype(int)
    df_feat["quarter"] = df_feat.index.quarter

    # Lag feature based on 4-week seasonality
    df_feat["lag_4"] = df_feat[target_col].shift(4)
    
    # Rolling mean of previous 4 weeks
    df_feat["rolling_mean_4"] = (
        df_feat[target_col].shift(1).rolling(window=4).mean()
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
    
    df_weekly = df_raw.resample("W").size().to_frame("call_count")
    df_weekly = df_weekly.asfreq("W", fill_value=0)
    
    if not df_weekly.empty:
        df_weekly = df_weekly.iloc[:-1]
    
    df_weekly.to_csv(DATA_SOURCE_PROCESSED)
    print(f"Preprocessed and saved '{DATA_SOURCE_PROCESSED}'. Total {len(df_weekly)} weeks.")
    
except Exception as e:
    print(f"Error during preprocessing: {e}")
    raise

# --- 2. Feature Engineering ---
print("Creating features from 100% of the weekly data...")
X, y = create_features(df_weekly, "call_count")

print(f"Training features shape: {X.shape}")
print(f"Training target shape: {y.shape}")

if X.empty:
    print("FATAL: No features were created. Data shorter than 4 weeks.")
else:
    # --- 3. Model Training ---
    print("Training XGBoost model on full weekly dataset...")
    
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

print("--- Weekly Training Script Finished ---")
