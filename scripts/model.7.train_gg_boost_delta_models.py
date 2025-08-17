# model.7.train_gg_boost_delta_models.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
MODEL_DIR = Path(__file__).resolve().parent / '../models'
DELTA_FILE = BASE_DATA_DIR / 'training_dataset_gg_boost_delta.csv'
SUB_FILE   = BASE_DATA_DIR / 'training_dataset_gg_sub.csv'

def _drop_constant_onehots(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = []
    for c in df.columns:
        if c.startswith(("role_", "bodytype_", "foot_", "accelerateType_", "chem_style_")):
            if df[c].nunique(dropna=False) <= 1:
                to_drop.append(c)
    return df.drop(columns=to_drop)

def _train_sub_model(df_subset: pd.DataFrame, target_col: str, role_name: str, model_suffix: str):
    if len(df_subset) < 50:
        print(f"Skipping '{role_name} - {model_suffix}' (only {len(df_subset)} samples).")
        return
    print(f"\n--- Training: {role_name} | {model_suffix} ({len(df_subset)} samples) ---")
    X = df_subset.drop(columns=[target_col]).copy()
    y = df_subset[target_col].astype(float)

    # Scale features + target; use non-negative coefficients for monotonicity
    feat_scaler = StandardScaler()
    Xs = pd.DataFrame(feat_scaler.fit_transform(X), columns=X.columns, index=X.index)
    targ_scaler = MinMaxScaler((0, 1))
    ys = targ_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    model = ElasticNetCV(l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.8, 1.0],
                         cv=5, random_state=42, n_jobs=-1, max_iter=5000,
                         positive=True).fit(Xs.values, ys)

    y_pred = targ_scaler.inverse_transform(model.predict(Xs.values).reshape(-1,1)).ravel()
    r2  = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"  R²: {r2:.4f} | RMSE: {rmse:.4f} | Non-zero Coefs: {(model.coef_!=0).sum()}/{len(model.coef_)}")

    # Unscaled-X, unscaled-y coefficients
    coef_unscaled = (model.coef_ / targ_scaler.scale_[0]) / feat_scaler.scale_

    safe_role_name = role_name.replace(" ", "_").replace("-", "_")
    model_filename = f"{safe_role_name}_{model_suffix}_model.pkl"
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_scaler": feat_scaler,
        "target_scaler": targ_scaler,
        "features": X.columns.tolist(),
        "coef_unscaled": coef_unscaled.tolist()
    }, MODEL_DIR / model_filename)
    print(f"  💾 Model saved to {model_filename}")

def _train_delta_model(df_subset: pd.DataFrame, target_col: str, role_name: str, model_suffix: str):
    if len(df_subset) < 50:
        print(f"Skipping '{role_name} - {model_suffix}' (only {len(df_subset)} samples).")
        return
    print(f"\n--- Training: {role_name} | {model_suffix} ({len(df_subset)} samples) ---")
    X = df_subset.drop(columns=[target_col]).copy().fillna(0)
    y = df_subset[target_col].astype(float)

    feat_scaler = StandardScaler()
    Xs = pd.DataFrame(feat_scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Keep LassoCV (deltas can be +/-); no target scaler needed here
    model = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000).fit(Xs.values, y.values)
    y_pred = model.predict(Xs.values)
    print(f"  R²: {r2_score(y, y_pred):.4f} | RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")

    safe_role_name = role_name.replace(" ", "_").replace("-", "_")
    model_filename = f"{safe_role_name}_{model_suffix}_model.pkl"
    joblib.dump({
        "model": model,
        "feature_scaler": feat_scaler,
        "target_scaler": None,                 # not used for delta
        "features": X.columns.tolist()
    }, MODEL_DIR / model_filename)
    print(f"  💾 Model saved to {model_filename}")

def _train_block(df: pd.DataFrame, target_col: str, suffix: str, drop_extra_onehots=True):
    role_cols = [c for c in df.columns if c.startswith('role_')]
    for role_col in role_cols:
        role_name = role_col.replace('role_', '').replace('_', ' ')
        df_subset = df[df[role_col] == 1].copy()
        if drop_extra_onehots:
            df_subset = _drop_constant_onehots(df_subset)
        if target_col not in df_subset.columns:
            print(f"⚠️ {target_col} not found for role '{role_name}'. Skipping.")
            continue
        if suffix == "ggMetaSub":
            _train_sub_model(df_subset, target_col, role_name, suffix)
        else:
            _train_delta_model(df_subset, target_col, role_name, suffix)

def main():
    print("🚀 Starting gg model training script...")
    MODEL_DIR.mkdir(exist_ok=True)

    # ggMetaSub (absolute)
    if SUB_FILE.exists():
        df_sub = pd.read_csv(SUB_FILE)
        _train_block(df_sub, target_col="target_ggMetaSub", suffix="ggMetaSub", drop_extra_onehots=True)
    else:
        print(f"❌ {SUB_FILE.name} not found. Run the build script first.")

    # ggMetaBoostDelta (delta to Basic)
    if DELTA_FILE.exists():
        df_delta = pd.read_csv(DELTA_FILE)
        _train_block(df_delta, target_col="target_ggMetaBoostDelta", suffix="ggMetaBoostDelta", drop_extra_onehots=True)
    else:
        print(f"❌ {DELTA_FILE.name} not found. Run the build script first.")

    print("\n🎉 All gg models trained successfully.")

if __name__ == "__main__":
    main()
