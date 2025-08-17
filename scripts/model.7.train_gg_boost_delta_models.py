# train_gg_boost_delta_models.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
MODEL_DIR = Path(__file__).resolve().parent / '../models'
DATA_DELTA_FILE = BASE_DATA_DIR / 'training_dataset_gg_boost_delta.csv'
DATA_SUB_FILE   = BASE_DATA_DIR / 'training_dataset_gg_sub.csv'

def _drop_constant_onehots(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = []
    for c in df.columns:
        if c.startswith(("role_", "bodytype_", "accelerateType_", "foot_")):
            if df[c].nunique(dropna=False) <= 1:
                to_drop.append(c)
    return df.drop(columns=to_drop)

def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c.startswith("attribute") or c in ("height", "weight", "skillMoves", "weakFoot", "familiarity"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# --------------------
# gg BOOST DELTA (as-is, with richer features)
# --------------------
def train_and_save_delta(df_subset, target_col, role_name, model_suffix):
    if len(df_subset) < 50:
        print(f"Skipping '{role_name} - {model_suffix}' (only {len(df_subset)} samples).")
        return

    print(f"\n--- Training (gg delta): {role_name} | {model_suffix} ({len(df_subset)} samples) ---")
    X = df_subset.drop(columns=[target_col]).fillna(0)
    y = df_subset[target_col]

    feature_scaler = StandardScaler()
    X_scaled = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)

    model = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000).fit(X_scaled.values, y.values)
    y_pred = model.predict(X_scaled.values)

    print(f"  R²: {r2_score(y, y_pred):.4f} | RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
    
    safe_role_name = role_name.replace(" ", "_").replace("-", "_")
    model_filename = f"{safe_role_name}_{model_suffix}_model.pkl"
    joblib.dump({
        "model": model,
        "feature_scaler": feature_scaler,
        "features": X.columns.tolist()
    }, MODEL_DIR / model_filename)
    print(f"  💾 Model saved to {model_filename}")

# --------------------
# ggMetaSub (absolute, non-negative, with unscaled coefs)
# --------------------
def train_and_save_sub(df_subset: pd.DataFrame, target_col: str, role_name: str, model_suffix: str):
    if len(df_subset) < 50:
        print(f"Skipping '{role_name} - {model_suffix}' (only {len(df_subset)} samples).")
        return

    print(f"\n--- Training (gg sub): {role_name} | {model_suffix} ({len(df_subset)} samples) ---")
    X = df_subset.drop(columns=[target_col]).copy()
    y = df_subset[target_col].astype(float)

    X = _clean_numeric(X).fillna(0)

    feature_scaler = StandardScaler()
    Xs = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    ys = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    model = ElasticNetCV(
        l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.8, 1.0],
        cv=5, random_state=42, n_jobs=-1, max_iter=5000, positive=True
    ).fit(Xs.values, ys)

    y_pred = target_scaler.inverse_transform(model.predict(Xs).reshape(-1, 1)).ravel()
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"  R²: {r2:.4f} | RMSE: {rmse:.4f} | NNZ: {(model.coef_ != 0).sum()}/{len(model.coef_)}")

    coef_unscaled = (model.coef_ / target_scaler.scale_[0]) / feature_scaler.scale_

    bundle = {
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "features": X.columns.tolist(),
        "coef_unscaled": coef_unscaled.tolist()
    }

    safe_role_name = role_name.replace(" ", "_").replace("-", "_")
    fn = f"{safe_role_name}_{model_suffix}_model.pkl"
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / fn)
    print(f"  💾 Model saved to {fn}")

def main():
    print("🚀 Starting gg model training script...")
    MODEL_DIR.mkdir(exist_ok=True)
    
    # --- BOOST DELTA ---
    if DATA_DELTA_FILE.exists():
        df_delta = pd.read_csv(DATA_DELTA_FILE)
        role_cols = [c for c in df_delta.columns if c.startswith('role_')]
        for role_col in role_cols:
            role_name = role_col.replace('role_', '').replace('_', ' ')
            df_subset = df_delta[df_delta[role_col] == 1].copy()
            df_subset = _drop_constant_onehots(df_subset)
            train_and_save_delta(df_subset, 'target_ggMetaBoostDelta', role_name, 'ggMetaBoostDelta')
    else:
        print(f"❌ {DATA_DELTA_FILE.name} not found. Skipping delta models.")

    # --- ggMetaSub (absolute) ---
    if DATA_SUB_FILE.exists():
        df_sub = pd.read_csv(DATA_SUB_FILE)
        role_cols = [c for c in df_sub.columns if c.startswith('role_')]
        for role_col in role_cols:
            role_name = role_col.replace('role_', '').replace('_', ' ')
            df_subset = df_sub[df_sub[role_col] == 1].copy()
            df_subset = _drop_constant_onehots(df_subset)
            train_and_save_sub(df_subset, 'target_ggMetaSub', role_name, 'ggMetaSub')
    else:
        print(f"❌ {DATA_SUB_FILE.name} not found. Skipping ggMetaSub models.")

    print("\n🎉 All gg models trained successfully.")

if __name__ == "__main__":
    main()
