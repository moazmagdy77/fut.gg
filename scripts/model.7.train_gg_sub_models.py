# model.7.train_gg_boost_delta_models.py
# Trains per-role ABSOLUTE ggMeta models which we will use to predict ggMetaSub
# by feeding "no-chem" (unboosted) features at inference.

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
MODEL_DIR = Path(__file__).resolve().parent / '../models'
DATA_FILE = BASE_DATA_DIR / 'training_dataset_gg_sub_abs.csv'  # built by step 6

def _drop_constant_onehots(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = []
    for c in df.columns:
        if c.startswith(("role_", "bodytype_", "accelerateType_", "foot_")):
            if df[c].nunique(dropna=False) <= 1:
                to_drop.append(c)
    return df.drop(columns=to_drop)

def train_and_save_model(df_subset, target_col, role_name, model_suffix):
    if len(df_subset) < 80:
        print(f"Skipping '{role_name} - {model_suffix}' (only {len(df_subset)} samples).")
        return

    print(f"\n--- Training: {role_name} | {model_suffix} ({len(df_subset)} samples) ---")
    X = df_subset.drop(columns=[target_col]).copy().fillna(0)
    y = df_subset[target_col].astype(float).values

    # Scale X, y
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = target_scaler.fit_transform(y.reshape(-1,1)).ravel()

    # Non-negative linear model (attributes monotonicity)
    model = ElasticNetCV(
        l1_ratio=[0.05,0.1,0.3,0.5,0.8,1.0],
        cv=5, random_state=42, n_jobs=-1, max_iter=5000, positive=True
    ).fit(X_scaled, y_scaled)

    # Evaluate back on original scale
    y_pred_scaled = model.predict(X_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    nnz = int((model.coef_ != 0).sum())
    print(f"  R²: {r2:.4f} | RMSE: {rmse:.4f} | Non-zero Coefs: {nnz}/{len(model.coef_)}")

    # Unscaled-X, unscaled-y coefficients (for delta math if ever needed)
    coef_unscaled = (model.coef_ / target_scaler.scale_[0]) / feature_scaler.scale_

    bundle = {
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "features": X.columns.tolist(),
        "coef_unscaled": coef_unscaled.tolist()
    }

    safe_role_name = role_name.replace(" ", "_").replace("-", "_")
    model_filename = f"{safe_role_name}_{model_suffix}_model.pkl"  # e.g., "LM_Inside_Forward_ggMetaSub_model.pkl"
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / model_filename)
    print(f"  💾 Model saved to {model_filename}")

def main():
    print("🚀 Training ABSOLUTE ggMeta models for ggMetaSub...")
    MODEL_DIR.mkdir(exist_ok=True)

    if not DATA_FILE.exists():
        print(f"❌ {DATA_FILE.name} not found. Run step 6 first.")
        return

    df = pd.read_csv(DATA_FILE)
    # Train PER ROLE (role_* one-hots exist)
    role_cols = [c for c in df.columns if c.startswith('role_')]
    if not role_cols:
        print("❌ No role_* columns found. Check dataset build.")
        return

    for role_col in role_cols:
        role_name = role_col.replace('role_', '').replace('_', ' ')
        df_subset = df[df[role_col] == 1].copy()
        df_subset = _drop_constant_onehots(df_subset)

        if "target_ggMetaAbs" not in df_subset.columns:
            print(f"⚠️ target_ggMetaAbs missing for role '{role_name}'. Skipping.")
            continue

        train_and_save_model(df_subset, target_col='target_ggMetaAbs', role_name=role_name, model_suffix='ggMetaSub')

    print("\n🎉 All ggMetaSub models trained successfully.")

if __name__ == "__main__":
    main()
