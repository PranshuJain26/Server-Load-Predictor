import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M59 (Naive, Weekly) Training ---")

MODEL_NAME = "A59\\M59_weekly.joblib"
DATA_SOURCE_RAW = "A59\\A59.csv"
DATA_SOURCE_PROCESSED = "A59\\A59_weekly_calls.csv"

# --------------------------------------------------
# 1. Preprocessing (WEEKLY)
# --------------------------------------------------
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")

df_raw = pd.read_csv(DATA_SOURCE_RAW)

df_raw["ds"] = pd.to_datetime(
    df_raw["Time of Call"],
    format="mixed",
    dayfirst=False
)

df_weekly = (
    df_raw
    .groupby(pd.Grouper(key="ds", freq="W"))
    .size()
    .to_frame("y")
)

df_weekly = (
    df_weekly
    .asfreq("W", fill_value=0)
    .reset_index()
)

# Drop last incomplete week
if len(df_weekly) > 1:
    df_weekly = df_weekly.iloc[:-1]

df_weekly.to_csv(DATA_SOURCE_PROCESSED, index=False)
print(f"Saved weekly data → {DATA_SOURCE_PROCESSED} ({len(df_weekly)} rows)")

# --------------------------------------------------
# 2. Train / Validation Split (80%)
# --------------------------------------------------
split = int(len(df_weekly) * 0.8)
train = df_weekly.iloc[:split]
test = df_weekly.iloc[split:]

# --------------------------------------------------
# 3. Naive Model (Last Value)
# --------------------------------------------------
last_value = train["y"].iloc[-1]
preds = np.repeat(last_value, len(test))

rmse = np.sqrt(mean_squared_error(test["y"].values, preds))
print(f"Validation RMSE: {rmse:.3f}")

# --------------------------------------------------
# 4. Save Model (last observed value)
# --------------------------------------------------
joblib.dump(last_value, MODEL_NAME)
print(f"Saved model → {MODEL_NAME}")

print("--- M59 Weekly Naive Training Finished ---")
