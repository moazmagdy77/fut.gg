# train_gg_delta_models.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
MODEL_DIR = Path(__file__).resolve().parent / '../models'
DATA_FILE = BASE_DATA_DIR / 'training_dataset_gg_delta.csv'

def train_and_save_model(df_subset, target_col, role_name, model_suffix):
    if len(df_subset) < 50:
        print(f"Skipping '{role_name} - {model_suffix}' (only {len(df_subset)} samples).")
        return

    print(f"\n--- Training: {role_name} | {model_suffix} ({len(df_subset)} samples) ---")
    X = df_subset.drop(columns=[target_col]).fillna(0)
    y = df_subset[target_col]

    feature_scaler = StandardScaler()
    X_scaled = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)

    model = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000).fit(X_scaled.values, y.values)
    y_pred = model.predict(X_scaled.values)

    print(f"  R² Score: {r2_score(y, y_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
    
    safe_role_name = role_name.replace(" ", "_").replace("-", "_")
    model_filename = f"{safe_role_name}_{model_suffix}_model.pkl"
    joblib.dump({
        "model": model, "feature_scaler": feature_scaler, "features": X.columns.tolist()
    }, MODEL_DIR / model_filename)
    print(f"  💾 Model saved to {model_filename}")

def main():
    print("🚀 Starting ggMeta DELTA model training script...")
    MODEL_DIR.mkdir(exist_ok=True)
    
    if not DATA_FILE.exists():
        print(f"❌ Error: {DATA_FILE.name} not found. Please run the build script first.")
        return
    
    df = pd.read_csv(DATA_FILE)
    role_cols = [col for col in df.columns if col.startswith('role_')]

    for role_col in role_cols:
        role_name = role_col.replace('role_', '').replace('_', ' ')
        df_subset = df[df[role_col] == 1].copy()
        
        cols_to_drop = [c for c in df_subset.columns if c.startswith(('role_', 'bodytype_', 'foot_'))]
        df_subset.drop(columns=cols_to_drop, inplace=True)
        
        train_and_save_model(df_subset, 'target_ggMetaDelta', role_name, 'ggMetaDelta')
        
    print("\n🎉 All ggMeta delta models trained successfully.")

if __name__ == "__main__":
    main()