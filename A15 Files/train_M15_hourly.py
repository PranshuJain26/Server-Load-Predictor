import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M15 (SARIMA, Hourly) Training ---")

MODEL_NAME = "A15\\M15_hourly.joblib"
DATA_SOURCE_RAW = "A15\\A15.csv"
DATA_SOURCE_PROCESSED = "A15\\A15_hourly_calls.csv"

# --------------------------------------------------
# 1. Preprocessing (HOURLY)
# --------------------------------------------------
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")

df_raw = pd.read_csv(DATA_SOURCE_RAW)

df_raw["datetime"] = pd.to_datetime(
    df_raw["Time of Call"],
    format="mixed",
    dayfirst=False
)

df_raw = df_raw.set_index("datetime")

# Hourly aggregation
df_hourly = df_raw.resample("H").size().to_frame("call_count")
df_hourly = df_hourly.asfreq("H", fill_value=0)

# Drop last incomplete hour
if not df_hourly.empty:
    df_hourly = df_hourly.iloc[:-1]

df_hourly.to_csv(DATA_SOURCE_PROCESSED)
print(f"Saved hourly data → {DATA_SOURCE_PROCESSED} ({len(df_hourly)} rows)")

# --------------------------------------------------
# 2. Train / Validation Split (80%)
# --------------------------------------------------
y = df_hourly["call_count"].values

split = int(len(y) * 0.8)
y_train, y_test = y[:split], y[split:]

# --------------------------------------------------
# 3. Train SARIMA Model (HOURLY)
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

print("--- M15 Hourly SARIMA Training Finished ---")
