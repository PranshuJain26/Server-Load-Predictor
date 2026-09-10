import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M42 (Prophet, Weekly) Training ---")

MODEL_NAME = "A42\\M42_weekly.joblib"
DATA_SOURCE_RAW = "A42\\A42.csv"
DATA_SOURCE_PROCESSED = "A42\\A42_weekly_calls.csv"

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

df_raw = df_raw.set_index("ds")

df_weekly = df_raw.resample("W").size().reset_index(name="y")

# Drop last incomplete week
if not df_weekly.empty:
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
# 3. Train Prophet Model
# --------------------------------------------------
print("Training Prophet model...")

model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=False
)

model.fit(train)

# --------------------------------------------------
# 4. Validation RMSE
# --------------------------------------------------
future = model.make_future_dataframe(periods=len(test), freq="W")
forecast = model.predict(future)

preds = forecast["yhat"].iloc[-len(test):].values
rmse = np.sqrt(mean_squared_error(test["y"].values, preds))

print(f"Validation RMSE (Weekly Prophet): {rmse:.3f}")

# --------------------------------------------------
# 5. Save Model
# --------------------------------------------------
joblib.dump(model, MODEL_NAME)
print(f"Saved model → {MODEL_NAME}")

print("--- M42 Weekly Prophet Training Finished ---")
