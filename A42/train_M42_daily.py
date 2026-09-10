import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M42 (Naive, Daily) Training ---")

MODEL_NAME = "A42\\M42_daily.joblib"
DATA_SOURCE_RAW = "A42\\A42.csv"
DATA_SOURCE_PROCESSED = "A42\\A42_daily_calls.csv"

# --------------------------------------------------
# 1. Preprocessing (DAILY)
# --------------------------------------------------
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")

df_raw = pd.read_csv(DATA_SOURCE_RAW)

df_raw["datetime"] = pd.to_datetime(
    df_raw["Time of Call"],
    format="mixed",
    dayfirst=False
)

df_raw = df_raw.set_index("datetime")

df_daily = df_raw.resample("D").size().to_frame("call_count")
df_daily = df_daily.asfreq("D", fill_value=0)

# Drop last incomplete day
if not df_daily.empty:
    df_daily = df_daily.iloc[:-1]

df_daily.to_csv(DATA_SOURCE_PROCESSED)
print(f"Saved daily data → {DATA_SOURCE_PROCESSED} ({len(df_daily)} rows)")

# --------------------------------------------------
# 2. Train / Validation Split (80%)
# --------------------------------------------------
y = df_daily["call_count"].values
split = int(len(y) * 0.8)
y_train, y_test = y[:split], y[split:]

# --------------------------------------------------
# 3. Naive Forecast (UNCHANGED LOGIC)
# --------------------------------------------------
preds = np.repeat(y_train[-1], len(y_test))
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"Validation RMSE (Daily Naive): {rmse:.3f}")

# --------------------------------------------------
# 4. Save Model Artifact
# --------------------------------------------------
artifact = {
    "last_value": int(y_train[-1])
}

joblib.dump(artifact, MODEL_NAME)
print(f"Saved model artifact → {MODEL_NAME}")

print("--- M42 Daily Naive Training Finished ---")
