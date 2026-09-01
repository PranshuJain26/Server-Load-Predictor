import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M15 (SARIMA, Weekly) Training ---")

MODEL_NAME = "A15\\M15_weekly.joblib"
DATA_SOURCE_RAW = "A15\\A15.csv"
DATA_SOURCE_PROCESSED = "A15\\A15_weekly_calls.csv"

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

df_raw = df_raw.set_index("datetime")

# Weekly aggregation
df_weekly = df_raw.resample("W").size().to_frame("call_count")
df_weekly = df_weekly.asfreq("W", fill_value=0)

# Drop last incomplete week
if not df_weekly.empty:
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
# 3. Train SARIMA Model (WEEKLY)
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

print("--- M15 Weekly SARIMA Training Finished ---")
