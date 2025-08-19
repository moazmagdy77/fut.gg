import os
import joblib
import pandas as pd
from pathlib import Path

# Define paths
base_dir = Path(__file__).resolve().parents[1]
data_dir = base_dir / "data"
model_dir = base_dir / "models"

PREDICTION_FILE = data_dir / "prediction_ready.csv"
OUTPUT_FILE = data_dir / "predicted_metaratings.csv"

# Load data
df = pd.read_csv(PREDICTION_FILE)

# Columns to ignore during prediction
ignore_cols = ['eaId', 'commonName', 'chemstyle', "position"]

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

# Append predictions
df['predicted_meta_rating'] = predictions

# Keep only two rows per (eaId, archetype): 
# 1. Chemstyle == "none"
# 2. Highest predicted meta rating (best chemstyle)
keep_rows = []

for (eaId, archetype), group in df.groupby(["eaId", "archetype"]):
    # Row with chemstyle == none
    none_row = group[group["chemstyle"] == "none"]
    if not none_row.empty:
        keep_rows.append(none_row.iloc[0])
    
    # Row with highest predicted meta rating (skip NaNs)
    valid_preds = group.dropna(subset=["predicted_meta_rating"])
    if not valid_preds.empty:
        best_row = valid_preds.loc[valid_preds["predicted_meta_rating"].idxmax()]
        keep_rows.append(best_row)

# Create final dataframe
final_df = pd.DataFrame(keep_rows)

# Save output
final_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Prediction complete. Results saved to {OUTPUT_FILE}")