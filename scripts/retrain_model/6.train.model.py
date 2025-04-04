import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# Define paths
base_dir = Path(__file__).resolve().parents[2]
data_path = base_dir / "data" / "training_dataset.csv"
model_dir = Path(__file__).resolve().parent / "models"

# Load preprocessed dataset
df = pd.read_csv(data_path)

# Ensure model output directory exists
os.makedirs(model_dir, exist_ok=True)

# Get archetypes
archetypes = df["archetype"].unique()

# Train model for each archetype
for arch in archetypes:
    subset = df[df["archetype"] == arch]
    if len(subset) < 20:
        print(f"Skipping '{arch}' due to low sample size ({len(subset)})")
        continue

    X = subset.drop(columns=["metaRating", "archetype"])
    y = subset["metaRating"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    model = LassoCV(cv=5, random_state=42, max_iter=10000).fit(X_scaled, y_scaled)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Save model and metadata
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "features": X.columns.tolist()
    }, model_dir / f"{arch}_lasso.pkl")

    # Show model stats
    print(f"\n✅ Trained model for archetype: {arch}")
    print(f"R² Score: {r2_score(y, y_pred):.4f}")
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"RMSE: {rmse:.4f}")
    print(f"Non-zero Coefficients: {(model.coef_ != 0).sum()} / {len(model.coef_)}")