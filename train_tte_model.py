#!/usr/bin/env python3
"""
Train an XGBoost regression model on TTE data.

Usage:
    python train_tte_model.py

Requirements:
    pip install xgboost scikit-learn joblib pandas
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import joblib

INPUT_CSV = "tte_data.csv"
OUTPUT_MODEL = "tte_model.pkl"
RANDOM_STATE = 42

def main() -> None:
    # 1. load data
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # 2. drop rows with any null value
    before = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {before - len(df)} rows with null values")

    # 3. compute ratio and filter outliers
    df["ratio"] = df["google_maps_seconds"] / df["graphhopper_seconds"]
    df = df[(df["ratio"] >= 0.3) & (df["ratio"] <= 3.0)]
    print(f"After ratio filter: {len(df)} rows")

    # 4. feature engineering
    #    target will be google_maps_seconds (already present)
    df["is_peak"] = (df["hour"].isin([8, 9, 17, 18])).astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    predictors = [
        "graphhopper_seconds",
        "distance_meters",
        "hour",
        "weekday",
        "is_peak",
        "is_weekend",
    ]
    X = df[predictors]
    y = df["google_maps_seconds"]

    # 5. train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 6. train XGBoost regressor
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # 7. evaluate
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # as %
    r2 = r2_score(y_test, y_pred)

    print(f"\nEvaluation on test set:")
    print(f"  MAE  : {mae:.1f} seconds")
    print(f"  MAPE : {mape:.1f} %")
    print(f"  R²   : {r2:.4f}")

    # 8. save model
    joblib.dump(model, OUTPUT_MODEL)
    print(f"\nModel saved to {OUTPUT_MODEL}")

    # 9. feature importance
    importance = model.feature_importances_
    print("\nFeature importance (gain):")
    for name, imp in sorted(
        zip(predictors, importance), key=lambda x: x[1], reverse=True
    ):
        print(f"  {name}: {imp:.4f}")


if __name__ == "__main__":
    main()
