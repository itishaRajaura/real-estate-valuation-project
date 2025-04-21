import argparse
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Pre-fetched dataset
real_estate_valuation = fetch_ucirepo(id=477)
X = real_estate_valuation.data.features
y = real_estate_valuation.data.targets
print(real_estate_valuation.metadata)
print(real_estate_valuation.variables)

def get_models():
    """Return a dictionary of candidate model pipelines."""
    return {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(random_state=42, n_estimators=100))
        ]),
        'GradientBoosting': Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingRegressor(random_state=42, n_estimators=100))
        ])
    }

def evaluate_and_select(X, y):
    """Cross-validate models and return the best one."""
    models = get_models()
    rmse_scores = {}
    for name, pipe in models.items():
        neg_mse = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
        rmse = np.sqrt(-neg_mse)
        rmse_scores[name] = rmse
        print(f"{name:17} CV RMSE = {rmse:.4f}")
    best_name = min(rmse_scores, key=rmse_scores.get)
    print(f"\n→ Best model: {best_name}")
    return best_name, models[best_name]

def train_and_save(model_path='best_model.joblib'):
    """Train and persist the best model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Evaluating models with 5‑fold CV:")
    best_name, best_pipe = evaluate_and_select(X_train, y_train)

    print(f"\nTraining final {best_name} on full train set…")
    best_pipe.fit(X_train, y_train)
    joblib.dump(best_pipe, model_path, compress=3)
    print(f"Saved best model to: {model_path}")

    y_pred = best_pipe.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"\nHold‑out Test RMSE = {rmse_test:.4f}")
    print(f"Hold‑out Test R²    = {r2:.4f}")

if _name_ == '_main_':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default='best_model.joblib')
    args = parser.parse_args()
    train_and_save(args.model_path)
