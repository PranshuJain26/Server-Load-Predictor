import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M69 (SARIMA, Hourly) Training ---")

MODEL_NAME = "A69\\M69_hourly.joblib"
DATA_SOURCE_RAW = "A69\\A69.csv"
DATA_SOURCE_PROCESSED = "A69\\A69_hourly_calls.csv"

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

df_hourly = (
    df_raw
    .groupby(pd.Grouper(key="datetime", freq="H"))
    .size()
    .to_frame("call_count")
)

df_hourly = df_hourly.asfreq("H", fill_value=0)

# Drop last incomplete hour
if len(df_hourly) > 1:
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
# 3. Train SARIMA Model
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
forecast = model.forecast(len(y_test))
rmse = np.sqrt(mean_squared_error(y_test, forecast))

print(f"Validation RMSE: {rmse:.3f}")

# --------------------------------------------------
# 5. Save Model
# --------------------------------------------------
joblib.dump(model, MODEL_NAME)
print(f"Saved model → {MODEL_NAME}")

print("--- M69 Hourly SARIMA Training Finished ---")
