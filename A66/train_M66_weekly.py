import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M66 (Moving Average, Weekly) Training ---")

MODEL_NAME = "A66\\M66_weekly.joblib"
DATA_SOURCE_RAW = "A66\\A66.csv"
DATA_SOURCE_PROCESSED = "A66\\A66_weekly_calls.csv"

WINDOW = 3  # moving average window

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

df_weekly = (
    df_raw
    .groupby(pd.Grouper(key="datetime", freq="W"))
    .size()
    .to_frame("call_count")
)

df_weekly = (
    df_weekly
    .asfreq("W", fill_value=0)
)

# Drop last incomplete week
if len(df_weekly) > 1:
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
# 4. Save Model (window + last history)
# --------------------------------------------------
joblib.dump(
    {
        "window": WINDOW,
        "history": list(y)
    },
    MODEL_NAME
)

print(f"Saved model → {MODEL_NAME}")
print("--- M66 Weekly Moving Average Training Finished ---")
