# model.5.train_es_models.py

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
DATA_0_CHEM_FILE = BASE_DATA_DIR / 'training_dataset_0_chem.csv'
DATA_3_CHEM_FILE = BASE_DATA_DIR / 'training_dataset_3_chem.csv'

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c.startswith("attribute") or c in ("height", "weight", "skillMoves", "weakFoot", "familiarity"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def train_and_save_model(df_subset: pd.DataFrame, target_col: str, role_name: str, model_suffix: str):
    if len(df_subset) < 50:
        print(f"Skipping '{role_name} - {model_suffix}' due to insufficient data ({len(df_subset)} samples).")
        return

    print(f"\n--- Training: {role_name} | {model_suffix} ({len(df_subset)} samples) ---")

    X = df_subset.drop(columns=[target_col]).copy()
    y = df_subset[target_col].astype(float)

    # Ensure numeric features only
    X = _clean_columns(X).fillna(0)
    non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        print(f"  ⚠️ Dropping non-numeric columns: {non_num[:6]}{'...' if len(non_num) > 6 else ''}")
        X = X.drop(columns=non_num)

    # Scale features & target
    feature_scaler = StandardScaler()
    X_scaled = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Non-negative linear model
    model = ElasticNetCV(
        l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.8, 1.0],
        cv=5, random_state=42, n_jobs=-1, max_iter=5000, positive=True
    ).fit(X_scaled.values, y_scaled)

    # Evaluate
    y_pred_scaled = model.predict(X_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    r2 = r2_score(y, y_pred); rmse = np.sqrt(mean_squared_error(y, y_pred))
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
    df = pd.read_csv(data_file)
    role_cols = [c for c in df.columns if c.startswith('role_')]
    for role_col in role_cols:
        role_name = role_col.replace('role_', '').replace('_', ' ')
        df_subset = df[df[role_col] == 1].copy()
        df_subset = _drop_constant_onehots(df_subset)
        if target_col not in df_subset.columns:
            print(f"⚠️ {target_col} not found for role '{role_name}'. Skipping.")
            continue
        train_and_save_model(df_subset, target_col, role_name, model_suffix)

def main():
    print("🚀 Starting model training script...")
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
