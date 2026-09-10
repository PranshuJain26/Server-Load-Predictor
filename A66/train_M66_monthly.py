import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M66 (Naive, Monthly) Training ---")

MODEL_NAME = "A66\\M66_monthly.joblib"
DATA_SOURCE_RAW = "A66\\A66.csv"
DATA_SOURCE_PROCESSED = "A66\\A66_monthly_calls.csv"

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

df_monthly = (
    df_monthly
    .asfreq("M", fill_value=0)
)

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
# 3. Naive Model (Last Observed Value)
# --------------------------------------------------
last_value = y_train[-1]
preds = np.repeat(last_value, len(y_test))

rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Validation RMSE: {rmse:.3f}")

# --------------------------------------------------
# 4. Save Model (last value)
# --------------------------------------------------
joblib.dump(last_value, MODEL_NAME)
print(f"Saved model → {MODEL_NAME}")

print("--- M66 Monthly Naive Training Finished ---")
