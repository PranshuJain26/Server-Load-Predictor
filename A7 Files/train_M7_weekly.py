import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M7 (Weekly Naive) Training ---")

MODEL_NAME = "A7\\M7_weekly.joblib"
DATA_SOURCE_RAW = "A7\\A7.csv"
DATA_SOURCE_PROCESSED = "A7\\A7_weekly_calls.csv"

# --------------------------------------------------
# 1. Preprocessing (WEEKLY)
# --------------------------------------------------
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")

df_raw = pd.read_csv(DATA_SOURCE_RAW)

df_raw["datetime"] = pd.to_datetime(
    df_raw["Time of Call"],
    format="mixed",
    dayfirst=False
)

df_raw = df_raw.set_index("datetime")

# Weekly aggregation (Mon–Sun)
df_weekly = df_raw.resample("W").size().to_frame("call_count")
df_weekly = df_weekly.asfreq("W", fill_value=0)

# Drop last incomplete week
if not df_weekly.empty:
    df_weekly = df_weekly.iloc[:-1]

df_weekly.to_csv(DATA_SOURCE_PROCESSED)
print(f"Saved weekly data → {DATA_SOURCE_PROCESSED} ({len(df_weekly)} weeks)")

# --------------------------------------------------
# 2. Train / Validation Split
# --------------------------------------------------
y = df_weekly["call_count"].values

split = int(len(y) * 0.8)
y_train, y_test = y[:split], y[split:]

# --------------------------------------------------
# 3. Weekly Naive Forecast (lag = 1 week)
# --------------------------------------------------
if len(y_train) < 7:
    raise ValueError("Need at least 7 weeks of training data")

last_week_value = y_train[-7]
predictions = np.repeat(last_week_value, len(y_test))

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Validation RMSE (Weekly Naive): {rmse:.2f}")

# --------------------------------------------------
# 4. Save Naive Artifact (just last values)
# --------------------------------------------------
artifact = {
    "last_week_value": last_week_value,
    "train_length": len(y)
}

joblib.dump(artifact, MODEL_NAME)
print(f"Saved naive model artifact → {MODEL_NAME}")

print("--- M7 Weekly Naive Training Finished ---")
