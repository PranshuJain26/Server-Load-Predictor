import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M7 (Prophet, Hourly) Training ---")

MODEL_NAME = "A7\\M7_hourly.joblib"
DATA_SOURCE_RAW = "A7\\A7.csv"
DATA_SOURCE_PROCESSED = "A7\\A7_hourly_calls.csv"

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
# 2. Prophet Dataset
# --------------------------------------------------
df_prophet = (
    df_hourly
    .reset_index()
    .rename(columns={"datetime": "ds", "call_count": "y"})
)

# --------------------------------------------------
# 3. Train / Validation Split (80%)
# --------------------------------------------------
split = int(len(df_prophet) * 0.8)
train = df_prophet.iloc[:split]
test = df_prophet.iloc[split:]

# --------------------------------------------------
# 4. Train Prophet Model (HOURLY)
# --------------------------------------------------
print("Training Prophet model...")

model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
)

model.fit(train)

# --------------------------------------------------
# 5. Validation RMSE
# --------------------------------------------------
future = model.make_future_dataframe(
    periods=len(test),
    freq="H"
)

forecast = model.predict(future)
preds = forecast["yhat"].iloc[-len(test):].values

rmse = np.sqrt(mean_squared_error(test["y"].values, preds))
print(f"Validation RMSE: {rmse:.2f}")

# --------------------------------------------------
# 6. Save Model
# --------------------------------------------------
joblib.dump(model, MODEL_NAME)
print(f"Saved model → {MODEL_NAME}")

print("--- M7 Hourly Prophet Training Finished ---")
