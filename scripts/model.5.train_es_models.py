# model.5.train_es_models.py

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# --- Dependency Check ---
try:
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import r2_score, mean_squared_error
except ImportError:
    print("❌ Critical Error: 'scikit-learn' is not installed.")
    print("👉 Run: pip install scikit-learn")
    sys.exit(1)

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
MODEL_DIR = Path(__file__).resolve().parent / '../models'
DATA_0_CHEM_FILE = BASE_DATA_DIR / 'training_dataset_0_chem.csv'
DATA_3_CHEM_FILE = BASE_DATA_DIR / 'training_dataset_3_chem.csv'

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Force core attributes to numeric, coercing errors to NaN
    for c in df.columns:
        if c.startswith("attribute") or c in ("height", "weight", "skillMoves", "weakFoot", "familiarity"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def train_and_save_model(df_subset: pd.DataFrame, target_col: str, role_name: str, model_suffix: str):
    # Initial size check
    if len(df_subset) < 50:
        print(f"Skipping '{role_name} - {model_suffix}' due to insufficient data ({len(df_subset)} samples).")
        return

    print(f"\n--- Training: {role_name} | {model_suffix} ---")

    X = df_subset.drop(columns=[target_col]).copy()
    y = df_subset[target_col].astype(float)

    # 1. Clean and Drop NAs (Don't fill with 0, drop bad rows)
    X = _clean_columns(X)
    
    # Align X and y by index after cleaning (in case dropping happened? usually dropna happens next)
    # We combine them temporarily to drop rows safely
    combined = pd.concat([X, y.rename('target')], axis=1).dropna()
    
    if combined.empty or len(combined) < 10:
        print(f"  ⚠️ Skipping: Dataset became empty after dropping NaNs.")
        return

    X = combined.drop(columns=['target'])
    y = combined['target']

    # 2. Drop non-numeric columns that might have slipped through
    non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        X = X.drop(columns=non_num)

    # Scale features
    feature_scaler = StandardScaler()
    X_scaled = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Scale target
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    # 3. Train Model (positive=False allows negative weights like Height penalties)
    try:
        model = ElasticNetCV(
            l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.8, 1.0],
            cv=5, random_state=42, n_jobs=-1, max_iter=5000, 
            positive=False 
        ).fit(X_scaled.values, y_scaled)
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        return

    # Evaluate
    y_pred_scaled = model.predict(X_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    nnz = int((model.coef_ != 0).sum())
    print(f"  R²: {r2:.4f} | RMSE: {rmse:.4f} | Non-zero Coefs: {nnz}/{len(model.coef_)}")

    # Unscaled coefficients for anchored deltas
    coef_unscaled = (model.coef_ / target_scaler.scale_[0]) / feature_scaler.scale_

    model_bundle = {
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "features": X.columns.tolist(),
        "coef_unscaled": coef_unscaled.tolist()
    }

    safe_role_name = role_name.replace(" ", "_").replace("-", "_")
    model_filename = f"{safe_role_name}_{model_suffix}_model.pkl"
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model_bundle, MODEL_DIR / model_filename)
    print(f"  💾 Model saved to {model_filename}")

def _drop_constant_onehots(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = []
    for c in df.columns:
        if c.startswith(("role_", "bodytype_", "accelerateType_", "foot_")):
            if df[c].nunique(dropna=False) <= 1:
                to_drop.append(c)
    return df.drop(columns=to_drop)

def _train_block(data_file: Path, target_col: str, model_suffix: str):
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"❌ Error reading {data_file.name}: {e}")
        return

    role_cols = [c for c in df.columns if c.startswith('role_')]
    for role_col in role_cols:
        role_name = role_col.replace('role_', '').replace('_', ' ')
        df_subset = df[df[role_col] == 1].copy()
        df_subset = _drop_constant_onehots(df_subset)
        if target_col not in df_subset.columns:
            continue
        train_and_save_model(df_subset, target_col, role_name, model_suffix)

def main():
    print("🚀 Starting model training script (Step 5)...")
    MODEL_DIR.mkdir(exist_ok=True)

    print("\n" + "="*50)
    print(" L O A D I N G   0 - C H E M   D A T A S E T")
    print("="*50)
    if DATA_0_CHEM_FILE.exists():
        _train_block(DATA_0_CHEM_FILE, target_col='target_esMetaSub', model_suffix='esMetaSub')
    else:
        print(f"❌ {DATA_0_CHEM_FILE.name} not found. Skipping 0-chem models.")

    print("\n" + "="*50)
    print(" L O A D I N G   3 - C H E M   D A T A S E T")
    print("="*50)
    if DATA_3_CHEM_FILE.exists():
        _train_block(DATA_3_CHEM_FILE, target_col='target_esMeta', model_suffix='esMeta')
    else:
        print(f"❌ {DATA_3_CHEM_FILE.name} not found. Skipping 3-chem models.")

    print("\n🎉 All models trained successfully.")

if __name__ == "__main__":
    main()