import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M69 (Moving Average, Monthly) Training ---")

MODEL_NAME = "A69\\M69_monthly.joblib"
DATA_SOURCE_RAW = "A69\\A69.csv"
DATA_SOURCE_PROCESSED = "A69\\A69_monthly_calls.csv"

WINDOW = 3  # moving average window

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

df_monthly = (
    df_raw
    .groupby(pd.Grouper(key="datetime", freq="M"))
    .size()
    .to_frame("call_count")
)

df_monthly = df_monthly.asfreq("M", fill_value=0)

# Drop last incomplete month
if len(df_monthly) > 1:
    df_monthly = df_monthly.iloc[:-1]

df_monthly.to_csv(DATA_SOURCE_PROCESSED)
print(f"Saved monthly data → {DATA_SOURCE_PROCESSED} ({len(df_monthly)} rows)")

# --------------------------------------------------
# 2. Train / Validation Split (80%)
# --------------------------------------------------
y = df_monthly["call_count"].values
split = int(len(y) * 0.8)

y_train, y_test = y[:split], y[split:]

# --------------------------------------------------
# 3. Moving Average Model
# --------------------------------------------------
preds = []
history = list(y_train)

for _ in range(len(y_test)):
    preds.append(np.mean(history[-WINDOW:]))
    history.append(preds[-1])

rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Validation RMSE: {rmse:.3f}")

# --------------------------------------------------
# 4. Save Model (window + full history)
# --------------------------------------------------
joblib.dump(
    {
        "window": WINDOW,
        "history": list(y)
    },
    MODEL_NAME
)

print(f"Saved model → {MODEL_NAME}")
print("--- M69 Monthly Moving Average Training Finished ---")
