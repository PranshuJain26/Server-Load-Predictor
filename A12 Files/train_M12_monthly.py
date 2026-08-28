import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M12 (Moving Average, Monthly) Training ---")

MODEL_NAME = "A12\\M12_monthly.joblib"
DATA_SOURCE_RAW = "A12\\A12.csv"
DATA_SOURCE_PROCESSED = "A12\\A12_monthly_calls.csv"
WINDOW = 3  # same as your function

# --------------------------------------------------
# 1. Preprocessing (MONTHLY)
# --------------------------------------------------
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")

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

# Drop last incomplete month
if not df_monthly.empty:
    df_monthly = df_monthly.iloc[:-1]

df_monthly.to_csv(DATA_SOURCE_PROCESSED)
print(f"Saved monthly data → {DATA_SOURCE_PROCESSED} ({len(df_monthly)} months)")

# --------------------------------------------------
# 2. Train / Validation Split (80%)
# --------------------------------------------------
y = df_monthly["call_count"].values

split = int(len(y) * 0.8)
y_train, y_test = y[:split], y[split:]

# --------------------------------------------------
# 3. Monthly Moving Average Forecast (UNCHANGED LOGIC)
# --------------------------------------------------
preds = []
history = list(y_train)

for _ in range(len(y_test)):
    preds.append(np.mean(history[-WINDOW:]))
    history.append(preds[-1])

rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Validation RMSE (Monthly Moving Average): {rmse:.3f}")

# --------------------------------------------------
# 4. Save Model Artifact
# --------------------------------------------------
artifact = {
    "window": WINDOW,
    "history": list(y)
}

joblib.dump(artifact, MODEL_NAME)
print(f"Saved model artifact → {MODEL_NAME}")

print("--- M12 Monthly Moving Average Training Finished ---")
