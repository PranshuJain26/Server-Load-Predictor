import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M12 (XGBoost, Weekly) Training ---")

MODEL_NAME = "A12\\M12_weekly.joblib"
DATA_SOURCE_RAW = "A12\\A12.csv"
DATA_SOURCE_PROCESSED = "A12\\A12_weekly_calls.csv"

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
print(f"Saved weekly data → {DATA_SOURCE_PROCESSED} ({len(df_weekly)} weeks)")

# --------------------------------------------------
# 2. Train / Validation Split (80%)
# --------------------------------------------------
y = df_weekly["call_count"].values

split = int(len(y) * 0.8)
y_train, y_test = y[:split], y[split:]

# --------------------------------------------------
# 3. Time Index Feature (SAME LOGIC)
# --------------------------------------------------
X_train = np.arange(len(y_train)).reshape(-1, 1)
X_test = np.arange(
    len(y_train),
    len(y_train) + len(y_test)
).reshape(-1, 1)

# --------------------------------------------------
# 4. Train XGBoost Model (WEEKLY)
# --------------------------------------------------
print("Training XGBoost model...")

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 5. Validation RMSE
# --------------------------------------------------
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"Validation RMSE: {rmse:.3f}")

# --------------------------------------------------
# 6. Save Model
# --------------------------------------------------
joblib.dump(model, MODEL_NAME)
print(f"Saved model → {MODEL_NAME}")

print("--- M12 Weekly XGBoost Training Finished ---")
