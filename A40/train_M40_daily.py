import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M40 (Prophet, Daily) Training ---")

MODEL_NAME = "A40\\M40_daily.joblib"
DATA_SOURCE_RAW = "A40\\A40.csv"
DATA_SOURCE_PROCESSED = "A40\\A40_daily_calls.csv"

# --------------------------------------------------
# 1. Preprocessing (DAILY)
# --------------------------------------------------
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")

df_raw = pd.read_csv(DATA_SOURCE_RAW)

df_raw["ds"] = pd.to_datetime(
    df_raw["Time of Call"],
    format="mixed",
    dayfirst=False
)

df_raw = df_raw.set_index("ds")

df_daily = df_raw.resample("D").size().reset_index(name="y")

# Drop last incomplete day
if not df_daily.empty:
    df_daily = df_daily.iloc[:-1]

df_daily.to_csv(DATA_SOURCE_PROCESSED, index=False)
print(f"Saved daily data → {DATA_SOURCE_PROCESSED} ({len(df_daily)} rows)")

# --------------------------------------------------
# 2. Train / Validation Split (80%)
# --------------------------------------------------
split = int(len(df_daily) * 0.8)
train = df_daily.iloc[:split]
test = df_daily.iloc[split:]

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
future = model.make_future_dataframe(periods=len(test), freq="D")
forecast = model.predict(future)

preds = forecast["yhat"].iloc[-len(test):].values
rmse = np.sqrt(mean_squared_error(test["y"].values, preds))

print(f"Validation RMSE: {rmse:.3f}")

# --------------------------------------------------
# 5. Save Model
# --------------------------------------------------
joblib.dump(model, MODEL_NAME)
print(f"Saved model → {MODEL_NAME}")

print("--- M40 Daily Prophet Training Finished ---")
