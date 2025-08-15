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
DATA_GG_FILE = BASE_DATA_DIR / 'training_dataset_gg.csv'

def train_and_save_model(df_subset, target_col, role_name, model_suffix):
    """Trains a single model for a specific role and target, then saves it."""
    
    if len(df_subset) < 50: # Minimum samples to train a meaningful model
        print(f"Skipping '{role_name} - {model_suffix}' due to insufficient data ({len(df_subset)} samples).")
        return

    print(f"\n--- Training: {role_name} | {model_suffix} ({len(df_subset)} samples) ---")

    X = df_subset.drop(columns=[target_col])
    y = df_subset[target_col]

    # Fill any potential missing values just in case
    X = X.fillna(0)

    # Scale features
    feature_scaler = StandardScaler()
    X_scaled = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Scale target variable
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Train the model
    model = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000)
    model.fit(X_scaled.values, y_scaled)

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
        "features": X.columns.tolist()
    }
    
    # Sanitize role_name for filename
    safe_role_name = role_name.replace(" ", "_").replace("-", "_")
    model_filename = f"{safe_role_name}_{model_suffix}_model.pkl"
    joblib.dump(model_bundle, MODEL_DIR / model_filename)
    print(f"  💾 Model saved to {model_filename}")

def main():
    print("🚀 Starting ggMeta model training script...")
    MODEL_DIR.mkdir(exist_ok=True)

    print("\n" + "="*50)
    print(" L O A D I N G   g g M e t a   D A T A S E T")
    print("="*50)

    if not DATA_GG_FILE.exists():
        print(f"❌ Error: {DATA_GG_FILE.name} not found. Please run the build script first.")
        return
    
    df = pd.read_csv(DATA_GG_FILE)
    role_cols = [col for col in df.columns if col.startswith('role_')]

    if not role_cols:
        print("❌ Error: No 'role_' columns found in the dataset. Please check the build script.")
        return

    for role_col in role_cols:
        role_name = role_col.replace('role_', '').replace('_', ' ')
        df_subset = df[df[role_col] == 1].copy()
        
        # Drop all one-hot encoded columns as they are constant for the subset
        cols_to_drop = [c for c in df_subset.columns if c.startswith('role_') or c.startswith('bodytype_') or c.startswith('accelerateType_')]
        df_subset.drop(columns=cols_to_drop, inplace=True)
        
        train_and_save_model(df_subset, 'target_ggMeta', role_name, 'ggMeta')
        
    print("\n🎉 All ggMeta models trained successfully.")

if __name__ == "__main__":
    main()