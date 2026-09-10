import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M40 (MovingAverage, Weekly) Training ---")

MODEL_NAME = "A40\\M40_weekly.joblib"
DATA_SOURCE_RAW = "A40\\A40.csv"
DATA_SOURCE_PROCESSED = "A40\\A40_weekly_calls.csv"
WINDOW = 3

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

df_weekly = df_raw.resample("W").size().to_frame("call_count")
df_weekly = df_weekly.asfreq("W", fill_value=0)

# Drop last incomplete week
if not df_weekly.empty:
    df_weekly = df_weekly.iloc[:-1]

df_weekly.to_csv(DATA_SOURCE_PROCESSED)
print(f"Saved weekly data → {DATA_SOURCE_PROCESSED} ({len(df_weekly)} rows)")

# --------------------------------------------------
# 2. Train / Validation Split (80%)
# --------------------------------------------------
y = df_weekly["call_count"].values
split = int(len(y) * 0.8)
y_train, y_test = y[:split], y[split:]

# --------------------------------------------------
# 3. Moving Average Forecast
# --------------------------------------------------
history = list(y_train)
preds = []

for _ in range(len(y_test)):
    preds.append(np.mean(history[-WINDOW:]))
    history.append(preds[-1])

rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Validation RMSE (Weekly MovingAverage): {rmse:.3f}")

# --------------------------------------------------
# 4. Save Model Artifact
# --------------------------------------------------
artifact = {
    "window": WINDOW,
    "last_values": y_train[-WINDOW:].tolist()
}

joblib.dump(artifact, MODEL_NAME)
print(f"Saved model artifact → {MODEL_NAME}")

print("--- M40 Weekly MovingAverage Training Finished ---")
