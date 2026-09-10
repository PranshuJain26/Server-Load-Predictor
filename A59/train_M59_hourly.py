import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M59 (Prophet, Hourly) Training ---")

MODEL_NAME = "A59\\M59_hourly.joblib"
DATA_SOURCE_RAW = "A59\\A59.csv"
DATA_SOURCE_PROCESSED = "A59\\A59_hourly_calls.csv"

# --------------------------------------------------
# 1. Preprocessing (HOURLY)
# --------------------------------------------------
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")

df_raw = pd.read_csv(DATA_SOURCE_RAW)

df_raw["ds"] = pd.to_datetime(
    df_raw["Time of Call"],
    format="mixed",
    dayfirst=False
)

df_hourly = (
    df_raw
    .groupby(pd.Grouper(key="ds", freq="H"))
    .size()
    .to_frame("y")
)

# FIX: Proper hourly frequency handling
df_hourly = (
    df_hourly
    .asfreq("H", fill_value=0)
    .reset_index()
)

# Drop last incomplete hour
if len(df_hourly) > 1:
    df_hourly = df_hourly.iloc[:-1]

df_hourly.to_csv(DATA_SOURCE_PROCESSED, index=False)
print(f"Saved hourly data → {DATA_SOURCE_PROCESSED} ({len(df_hourly)} rows)")

# --------------------------------------------------
# 2. Train / Validation Split (80%)
# --------------------------------------------------
split = int(len(df_hourly) * 0.8)
train = df_hourly.iloc[:split]
test = df_hourly.iloc[split:]

if train["y"].notna().sum() < 2:
    raise ValueError("Not enough data points to train Prophet model.")

# --------------------------------------------------
# 3. Train Prophet Model
# --------------------------------------------------
print("Training Prophet model...")

model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
)

model.fit(train)

# --------------------------------------------------
# 4. Validation RMSE
# --------------------------------------------------
future = model.make_future_dataframe(
    periods=len(test),
    freq="H"
)

forecast = model.predict(future)
preds = forecast["yhat"].iloc[-len(test):].values

rmse = np.sqrt(mean_squared_error(test["y"].values, preds))
print(f"Validation RMSE: {rmse:.3f}")

# --------------------------------------------------
# 5. Save Model
# --------------------------------------------------
joblib.dump(model, MODEL_NAME)
print(f"Saved model → {MODEL_NAME}")

print("--- M59 Hourly Prophet Training Finished ---")
