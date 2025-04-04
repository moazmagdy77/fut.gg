import os
import joblib
import pandas as pd
from pathlib import Path

# Define paths
base_dir = Path(__file__).resolve().parents[2]
data_dir = base_dir / "data"
model_dir = base_dir / "scripts" / "retrain_model" / "models"

PREDICTION_FILE = data_dir / "prediction_ready.csv"
OUTPUT_FILE = data_dir / "predicted_metaratings.csv"

# Load data
df = pd.read_csv(PREDICTION_FILE)

# Columns to ignore during prediction
ignore_cols = ['eaId', 'commonName', 'meta_rating']

# Placeholder for predictions
predictions = []

# Predict row by row
for idx, row in df.iterrows():
    archetype = row['archetype']
    model_path = model_dir / f"{archetype}_lasso.pkl"

    if not model_path.exists():
        print(f"[WARN] Model for archetype '{archetype}' not found. Skipping player at index {idx}.")
        predictions.append(None)
        continue

    loaded_data = joblib.load(model_path)
    model = loaded_data["model"]
    scaler = loaded_data["scaler"]
    expected_features = loaded_data["features"]
    target_scaler = loaded_data["target_scaler"]

    # Prepare input row
    input_row = row.drop(ignore_cols, errors='ignore')
    input_row = input_row.reindex(expected_features, fill_value=0).to_frame().T
    input_row_scaled = pd.DataFrame(scaler.transform(input_row), columns=expected_features)

    pred_scaled = model.predict(input_row_scaled)[0]
    pred = target_scaler.inverse_transform([[pred_scaled]])[0][0]
    predictions.append(pred)

# Append predictions and export
df['predicted_meta_rating'] = predictions
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Prediction complete. Results saved to {OUTPUT_FILE}")