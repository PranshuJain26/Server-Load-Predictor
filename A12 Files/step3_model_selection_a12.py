import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------
# Utilities
# -----------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def split_series(series, test_ratio=0.2):
    split = int(len(series) * (1 - test_ratio))
    return series[:split], series[split:]


# -----------------------------------
# Model evaluators
# -----------------------------------
def naive(y_train, y_test):
    return rmse(y_test, np.repeat(y_train[-1], len(y_test)))


def moving_average(y_train, y_test, window=3):
    preds, history = [], list(y_train)
    for _ in range(len(y_test)):
        preds.append(np.mean(history[-window:]))
        history.append(preds[-1])
    return rmse(y_test, preds)


def random_forest(y_train, y_test):
    X_train = np.arange(len(y_train)).reshape(-1, 1)
    X_test = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    return rmse(y_test, model.predict(X_test))


def xgboost(y_train, y_test):
    X_train = np.arange(len(y_train)).reshape(-1, 1)
    X_test = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    return rmse(y_test, model.predict(X_test))


def sarima(y_train, y_test):
    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    return rmse(y_test, model.forecast(len(y_test)))


def prophet_model(df):
    df_p = df.rename(
        columns={"Time of Call": "ds", "call_count": "y"}
    )[["ds", "y"]]

    split = int(len(df_p) * 0.8)
    train, test = df_p.iloc[:split], df_p.iloc[split:]

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    model.fit(train)

    future = model.make_future_dataframe(
        periods=len(test),
        freq=None
    )
    forecast = model.predict(future)

    preds = forecast["yhat"].iloc[-len(test):].values
    return rmse(test["y"].values, preds)


# -----------------------------------
# Main execution (A12)
# -----------------------------------
models = {
    "Naive": naive,
    "MovingAverage": moving_average,
    "RandomForest": random_forest,
    "XGBoost": xgboost,
    "SARIMA": sarima,
    "Prophet": prophet_model
}

model_rmse = {m: [] for m in models}

for granularity in ["hourly", "daily", "weekly", "monthly"]:
    print(f"\n🔍 Processing A12 {granularity}")

    df = pd.read_csv(f"A12/A12_{granularity}.csv")
    df["Time of Call"] = pd.to_datetime(df["Time of Call"])
    df = df.sort_values("Time of Call")

    series = df["call_count"].values

    if len(series) < 30:
        print("⚠️ Skipped (insufficient data)")
        continue

    y_train, y_test = split_series(series)

    for model_name, model_func in models.items():
        try:
            if model_name == "Prophet":
                score = model_func(df)
            else:
                score = model_func(y_train, y_test)

            model_rmse[model_name].append(score)
        except Exception:
            model_rmse[model_name].append(np.inf)

# -----------------------------------
# Results
# -----------------------------------
for model, score in model_rmse.items():
    print(f"{model} | {np.round(score, 3)}")

overall_results = []
for model, scores in model_rmse.items():
    avg_rmse = np.mean(scores)
    overall_results.append({
        "api_code": "A12",
        "model": model,
        "overall_rmse": round(avg_rmse, 3)
    })

results_df = (
    pd.DataFrame(overall_results)
    .sort_values("overall_rmse")
)

# Save final result
results_df.to_csv("A12/A12_best_model.csv", index=False)

print("\n🏆 BEST MODEL FOR A12")
print(results_df.head(1))
