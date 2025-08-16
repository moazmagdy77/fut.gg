# train_models.py

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
DATA_0_CHEM_FILE = BASE_DATA_DIR / 'training_dataset_0_chem.csv'
DATA_3_CHEM_FILE = BASE_DATA_DIR / 'training_dataset_3_chem.csv'

def train_and_save_model(df_subset, target_col, role_name, model_suffix):
    """Trains a single model for a specific role and target, then saves it."""
    
    if len(df_subset) < 50: # Minimum samples to train a meaningful model
        print(f"Skipping '{role_name} - {model_suffix}' due to insufficient data ({len(df_subset)} samples).")
        return

    print(f"\n--- Training: {role_name} | {model_suffix} ({len(df_subset)} samples) ---")

    X = df_subset.drop(columns=[target_col])
    y = df_subset[target_col]

    # --- FIX: Add this line to handle any missing values ---
    X = X.fillna(0)

    # Scale features
    feature_scaler = StandardScaler()
    X_scaled = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Scale target variable (helps with model convergence)
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Train the model (non-negative weights to enforce monotonicity)
    model = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000, positive=True)
    model.fit(X_scaled.values, y_scaled)

    # Convert coefficients to *unscaled-y, unscaled-X* contribution weights:
    coef_unscaled = (model.coef_ / target_scaler.scale_[0]) / feature_scaler.scale_

    # Evaluate the model
    y_pred_scaled = model.predict(X_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Non-zero Coefficients: {(model.coef_ != 0).sum()} / {len(model.coef_)}")

    # Save the complete model bundle
    model_bundle = {
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "features": X.columns.tolist(),
        "coef_unscaled": coef_unscaled.tolist()
    }
    
    # Sanitize role_name for filename
    safe_role_name = role_name.replace(" ", "_").replace("-", "_")
    model_filename = f"{safe_role_name}_{model_suffix}_model.pkl"
    joblib.dump(model_bundle, MODEL_DIR / model_filename)
    print(f"  💾 Model saved to {model_filename}")

def main():
    print("🚀 Starting model training script...")
    MODEL_DIR.mkdir(exist_ok=True)

    # --- Train 0-Chem (esMetaSub) Models ---
    print("\n" + "="*50)
    print(" L O A D I N G   0 - C H E M   D A T A S E T")
    print("="*50)
    if not DATA_0_CHEM_FILE.exists():
        print(f"❌ Error: {DATA_0_CHEM_FILE.name} not found. Skipping 0-chem models.")
    else:
        df_0_chem = pd.read_csv(DATA_0_CHEM_FILE)
        role_cols_0 = [col for col in df_0_chem.columns if col.startswith('role_')]

        for role_col in role_cols_0:
            role_name = role_col.replace('role_', '').replace('_', ' ')
            df_subset = df_0_chem[df_0_chem[role_col] == 1].copy()
            
            # Drop all one-hot encoded columns as they are constant for the subset
            cols_to_drop = [c for c in df_subset.columns if c.startswith('role_') or c.startswith('bodytype_') or c.startswith('accelerateType_')]
            df_subset.drop(columns=cols_to_drop, inplace=True)
            
            train_and_save_model(df_subset, 'target_esMetaSub', role_name, 'esMetaSub')

    # --- Train 3-Chem (esMeta) Models ---
    print("\n" + "="*50)
    print(" L O A D I N G   3 - C H E M   D A T A S E T")
    print("="*50)
    if not DATA_3_CHEM_FILE.exists():
        print(f"❌ Error: {DATA_3_CHEM_FILE.name} not found. Skipping 3-chem models.")
    else:
        df_3_chem = pd.read_csv(DATA_3_CHEM_FILE)
        role_cols_3 = [col for col in df_3_chem.columns if col.startswith('role_')]

        for role_col in role_cols_3:
            role_name = role_col.replace('role_', '').replace('_', ' ')
            df_subset = df_3_chem[df_3_chem[role_col] == 1].copy()

            cols_to_drop = [c for c in df_subset.columns if c.startswith('role_') or c.startswith('bodytype_') or c.startswith('accelerateType_')]
            df_subset.drop(columns=cols_to_drop, inplace=True)
            
            train_and_save_model(df_subset, 'target_esMeta', role_name, 'esMeta')
        
    print("\n🎉 All models trained successfully.")

if __name__ == "__main__":
    main()