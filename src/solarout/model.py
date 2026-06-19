"""Train and evaluate the daily solar output regressors.

Two model variants are trained on the same target (SolarOutput_kWh_per_kWp):

- "full"  -- includes Sunshine and Cloud9am/Cloud3pm, i.e. assumes the user
  (or weather forecast) can supply a direct irradiance/cloudiness proxy.
- "basic" -- only the fields a generic weather forecast commonly provides
  (temperature, rainfall, humidity, pressure, wind, month, location), no
  sunshine/cloud. This lets the agent answer questions when the user only
  describes general conditions.

Evaluation uses a temporal split (train on Jan-Oct, test on Nov-Dec of the
selected year) to mimic forecasting unseen future dates rather than randomly
held-out days, which is the realistic deployment scenario ("what will
tomorrow's output be").
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solarout.config import DAILY_DATASET_CSV, MODELS_DIR

TARGET = "SolarOutput_kWh_per_kWp"
CATEGORICAL_FEATURES = ["Location"]
BASIC_NUMERIC_FEATURES = [
    "MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm", "WindSpeed9am", "WindSpeed3pm",
    "month_sin", "month_cos",
]
FULL_NUMERIC_FEATURES = BASIC_NUMERIC_FEATURES + ["Sunshine", "Cloud9am", "Cloud3pm"]

TEST_MONTHS = (11, 12)


def make_pipeline(numeric_features: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("location", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", numeric_features),
        ]
    )
    model = HistGradientBoostingRegressor(random_state=42, max_iter=300, early_stopping=True)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def train_variant(df: pd.DataFrame, name: str, numeric_features: list[str]):
    feature_cols = CATEGORICAL_FEATURES + numeric_features
    train = df[~df["Month"].isin(TEST_MONTHS)]
    test = df[df["Month"].isin(TEST_MONTHS)]

    pipe = make_pipeline(numeric_features)
    pipe.fit(train[feature_cols], train[TARGET])

    pred_train = pipe.predict(train[feature_cols])
    pred_test = pipe.predict(test[feature_cols])

    metrics_train = evaluate(train[TARGET], pred_train)
    metrics_test = evaluate(test[TARGET], pred_test)

    print(f"\n=== {name} model ({len(numeric_features)} numeric features) ===")
    print(f"  train: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics_train.items()))
    print(f"  test:  " + ", ".join(f"{k}={v:.4f}" for k, v in metrics_test.items()))

    importance = permutation_importance(
        pipe, test[feature_cols], test[TARGET], n_repeats=5, random_state=42, scoring="r2"
    )
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importance.importances_mean}
    ).sort_values("importance", ascending=False)

    return pipe, metrics_train, metrics_test, test, pred_test, importance_df


def plot_predicted_vs_actual(results: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, (test, pred_test, metrics_test)) in zip(axes, results.items()):
        ax.scatter(test[TARGET], pred_test, alpha=0.3, s=10)
        lims = [0, max(test[TARGET].max(), pred_test.max()) * 1.05]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Actual output (kWh/kWp/day)")
        ax.set_ylabel("Predicted output (kWh/kWp/day)")
        ax.set_title(f"{name}: R2={metrics_test['R2']:.3f}, RMSE={metrics_test['RMSE']:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved plot -> {out_path}")


def main() -> None:
    df = pd.read_csv(DAILY_DATASET_CSV)

    results = {}
    importances = {}
    pipelines = {}
    for name, numeric_features in [("full", FULL_NUMERIC_FEATURES), ("basic", BASIC_NUMERIC_FEATURES)]:
        pipe, metrics_train, metrics_test, test, pred_test, importance_df = train_variant(
            df, name, numeric_features
        )
        results[name] = (test, pred_test, metrics_test)
        importances[name] = importance_df
        pipelines[name] = pipe
        print(f"\ntop features ({name}):")
        print(importance_df.head(8).to_string(index=False))

    plot_predicted_vs_actual(results, MODELS_DIR / "predicted_vs_actual.png")

    residual_std = {
        name: float(np.std(test[TARGET].to_numpy() - pred_test))
        for name, (test, pred_test, _metrics) in results.items()
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipelines": pipelines,
            "feature_sets": {"full": FULL_NUMERIC_FEATURES, "basic": BASIC_NUMERIC_FEATURES},
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET,
            "test_months": TEST_MONTHS,
            "test_metrics": {name: metrics for name, (_t, _p, metrics) in results.items()},
            "residual_std": residual_std,
        },
        MODELS_DIR / "solar_output_models.joblib",
    )
    print(f"\nsaved models -> {MODELS_DIR / 'solar_output_models.joblib'}")


if __name__ == "__main__":
    main()
