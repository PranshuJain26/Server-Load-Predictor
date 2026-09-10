import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M59 (XGBoost, Monthly) Training ---")

MODEL_NAME = "A59\\M59_monthly.joblib"
DATA_SOURCE_RAW = "A59\\A59.csv"
DATA_SOURCE_PROCESSED = "A59\\A59_monthly_calls.csv"

# --------------------------------------------------
# 1. Preprocessing (MONTHLY)
# --------------------------------------------------
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")

df_raw = pd.read_csv(DATA_SOURCE_RAW)

df_raw["ds"] = pd.to_datetime(
    df_raw["Time of Call"],
    format="mixed",
    dayfirst=False
)

df_monthly = (
    df_raw
    .groupby(pd.Grouper(key="ds", freq="M"))
    .size()
    .to_frame("y")
)

df_monthly = (
    df_monthly
    .asfreq("M", fill_value=0)
    .reset_index()
)

# Drop last incomplete month
if len(df_monthly) > 1:
    df_monthly = df_monthly.iloc[:-1]

df_monthly.to_csv(DATA_SOURCE_PROCESSED, index=False)
print(f"Saved monthly data → {DATA_SOURCE_PROCESSED} ({len(df_monthly)} rows)")

# --------------------------------------------------
# 2. Train / Validation Split (80%)
# --------------------------------------------------
y = df_monthly["y"].values
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

print("--- M59 Monthly XGBoost Training Finished ---")
