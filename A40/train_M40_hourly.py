import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M40 (XGBoost, Hourly) Training ---")

MODEL_NAME = "A40\\M40_hourly.joblib"
DATA_SOURCE_RAW = "A40\\A40.csv"
DATA_SOURCE_PROCESSED = "A40\\A40_hourly_calls.csv"

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

X_train = np.arange(len(y_train)).reshape(-1, 1)
X_test = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)

# --------------------------------------------------
# 3. Train XGBoost Model
# --------------------------------------------------
print("Training XGBoost model...")

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# --------------------------------------------------
# 4. Validation RMSE
# --------------------------------------------------
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"Validation RMSE: {rmse:.3f}")

# --------------------------------------------------
# 5. Save Model
# --------------------------------------------------
joblib.dump(model, MODEL_NAME)
print(f"Saved model → {MODEL_NAME}")

print("--- M40 Hourly XGBoost Training Finished ---")
