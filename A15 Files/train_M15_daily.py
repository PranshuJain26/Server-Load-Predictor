import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M15 (SARIMA, Daily) Training ---")

MODEL_NAME = "A15\\M15_daily.joblib"
DATA_SOURCE_RAW = "A15\\A15.csv"
DATA_SOURCE_PROCESSED = "A15\\A15_daily_calls.csv"

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

# Daily aggregation
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
# 3. Train SARIMA Model (DAILY)
# --------------------------------------------------
print("Training SARIMA model...")

model = SARIMAX(
    y_train,
    order=(1, 1, 1),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

# --------------------------------------------------
# 4. Validation RMSE
# --------------------------------------------------
preds = model.forecast(len(y_test))
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"Validation RMSE: {rmse:.3f}")

# --------------------------------------------------
# 5. Save Model
# --------------------------------------------------
joblib.dump(model, MODEL_NAME)
print(f"Saved model → {MODEL_NAME}")

print("--- M15 Daily SARIMA Training Finished ---")
