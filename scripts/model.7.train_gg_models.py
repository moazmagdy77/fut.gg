# train_gg_models.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import os

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
MODEL_DIR = Path(__file__).resolve().parent / '../models'
DATA_GG_ON_CHEM_FILE = BASE_DATA_DIR / 'training_dataset_gg_on_chem.csv'
DATA_GG_BASIC_FILE = BASE_DATA_DIR / 'training_dataset_gg_basic.csv'

def train_and_save_model(df_subset, target_col, role_name, model_suffix):
    if len(df_subset) < 50:
        print(f"Skipping '{role_name} - {model_suffix}' due to insufficient data ({len(df_subset)} samples).")
        return

    print(f"\n--- Training: {role_name} | {model_suffix} ({len(df_subset)} samples) ---")
    X = df_subset.drop(columns=[target_col]).fillna(0)
    y = df_subset[target_col]

    feature_scaler = StandardScaler()
    X_scaled = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    model = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000).fit(X_scaled.values, y_scaled)
    y_pred = target_scaler.inverse_transform(model.predict(X_scaled.values).reshape(-1, 1)).ravel()

    print(f"  R² Score: {r2_score(y, y_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
    
    safe_role_name = role_name.replace(" ", "_").replace("-", "_")
    model_filename = f"{safe_role_name}_{model_suffix}_model.pkl"
    joblib.dump({
        "model": model, "feature_scaler": feature_scaler, "target_scaler": target_scaler, "features": X.columns.tolist()
    }, MODEL_DIR / model_filename)
    print(f"  💾 Model saved to {model_filename}")

def main():
    print("🚀 Starting ggMeta model training script...")
    MODEL_DIR.mkdir(exist_ok=True)

    # --- Train On-Chem (ggMeta) Models ---
    print("\n" + "="*50 + "\n L O A D I N G   ggMeta 'On-Chem' D A T A S E T\n" + "="*50)
    if DATA_GG_ON_CHEM_FILE.exists():
        df = pd.read_csv(DATA_GG_ON_CHEM_FILE)
        for role_col in [c for c in df.columns if c.startswith('role_')]:
            role_name = role_col.replace('role_', '').replace('_', ' ')
            df_subset = df[df[role_col] == 1].copy()
            df_subset.drop(columns=[c for c in df_subset.columns if c.startswith(('role_', 'bodytype_', 'accelerateType_', 'foot_'))], inplace=True)
            train_and_save_model(df_subset, 'target_ggMeta', role_name, 'ggMeta')
    else:
        print(f"❌ Error: {DATA_GG_ON_CHEM_FILE.name} not found.")

    # --- Train Basic-Chem (ggMetaSub) Models ---
    print("\n" + "="*50 + "\n L O A D I N G  ggMeta 'Baseline' D A T A S E T\n" + "="*50)
    if DATA_GG_BASIC_FILE.exists():
        df = pd.read_csv(DATA_GG_BASIC_FILE)
        for role_col in [c for c in df.columns if c.startswith('role_')]:
            role_name = role_col.replace('role_', '').replace('_', ' ')
            df_subset = df[df[role_col] == 1].copy()
            df_subset.drop(columns=[c for c in df_subset.columns if c.startswith(('role_', 'bodytype_', 'accelerateType_', 'foot_'))], inplace=True)
            train_and_save_model(df_subset, 'target_ggMetaSub', role_name, 'ggMetaSub')
    else:
        print(f"❌ Error: {DATA_GG_BASIC_FILE.name} not found.")
        
    print("\n🎉 All ggMeta models trained successfully.")

if __name__ == "__main__":
    main()