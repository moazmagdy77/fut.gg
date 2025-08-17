# train_es_models.py

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


# ---------- Delta-head helpers (monotone) ----------
def _monotone_cols(all_cols):
    """
    Columns that can plausibly improve with evolutions:
      - attributes
      - playstyle columns (these are the raw playstyle names in your dataset)
      - skillMoves / weakFoot
    We exclude one-hots that are fixed and/or not true upgrades:
      role_, bodytype_, accelerateType_, foot_, and static anthropometrics.
    """
    bad_prefixes = ('role_', 'bodytype_', 'accelerateType_', 'foot_')
    keep = [
        c for c in all_cols
        if not any(c.startswith(p) for p in bad_prefixes)
        and c not in ('height', 'weight')  # static; deltas will be 0
    ]
    return keep


def _fit_delta_head(X_df, y_series, random_state=42, max_pairs=40000):
    """
    Learn a non-negative mapping from positive feature increases (ΔX⁺) to Δrating.
    Uses ElasticNetCV(positive=True). Operates in UN-SCALED feature/target space.
    """
    rng = np.random.RandomState(random_state)
    cols = _monotone_cols(X_df.columns.tolist())

    Xm = X_df[cols].values.astype(float)
    yv = y_series.values.astype(float)

    n = len(yv)
    if n < 80:  # not enough to build a useful delta head
        return cols, np.zeros(len(cols)).tolist()

    # sample pairs (j > i) where y_j > y_i
    I = rng.randint(0, n, size=max_pairs)
    J = rng.randint(0, n, size=max_pairs)
    mask = yv[J] > yv[I]
    I, J = I[mask], J[mask]

    if len(I) < 2000:
        # Fall back to (nearly) all ordered pairs up to a cap
        ii, jj = np.where(
            (yv.reshape(-1, 1) < yv.reshape(1, -1))
            & (np.arange(n)[:, None] != np.arange(n)[None, :])
        )
        if len(ii) > max_pairs:
            sel = rng.choice(len(ii), size=max_pairs, replace=False)
            ii, jj = ii[sel], jj[sel]
        I, J = ii, jj

    if len(I) == 0:
        return cols, np.zeros(len(cols)).tolist()

    dX = Xm[J] - Xm[I]
    dX = np.maximum(0.0, dX)   # only positive deltas
    dy = (yv[J] - yv[I]).astype(float)

    delta = ElasticNetCV(
        l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.8, 1.0],
        cv=5, random_state=random_state, n_jobs=-1, max_iter=5000, positive=True
    ).fit(dX, dy)

    return cols, delta.coef_.tolist()


# ---------- Core training ----------
def train_and_save_model(df_subset, target_col, role_name, model_suffix):
    """Trains a single model for a specific role and target, then saves it."""
    if len(df_subset) < 50:  # Minimum samples to train a meaningful model
        print(f"Skipping '{role_name} - {model_suffix}' due to insufficient data ({len(df_subset)} samples).")
        return

    print(f"\n--- Training: {role_name} | {model_suffix} ({len(df_subset)} samples) ---")

    X = df_subset.drop(columns=[target_col]).fillna(0)
    y = df_subset[target_col]

    # Scale features
    feature_scaler = StandardScaler()
    X_scaled = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Scale target variable (helps with model convergence)
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Train the absolute model with non-negative weights
    model = ElasticNetCV(
        l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.8, 1.0],
        cv=5, random_state=42, n_jobs=-1, max_iter=5000, positive=True
    )
    model.fit(X_scaled.values, y_scaled)

    # Evaluate
    y_pred_scaled = model.predict(X_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Non-zero Coefficients: {(model.coef_ != 0).sum()} / {len(model.coef_)}")

    # Convert coefficients to *unscaled-y, unscaled-X* contribution weights
    coef_unscaled = (model.coef_ / target_scaler.scale_[0]) / feature_scaler.scale_

    # Fit the monotone delta head in unscaled space
    delta_features, delta_coef = _fit_delta_head(X.fillna(0), y)

    # Save the complete model bundle (keep all artifacts)
    model_bundle = {
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "features": X.columns.tolist(),
        "coef_unscaled": coef_unscaled.tolist(),
        "delta_features": delta_features,
        "delta_coef": delta_coef
    }

    # Sanitize role_name for filename
    safe_role_name = role_name.replace(" ", "_").replace("-", "_")
    model_filename = f"{safe_role_name}_{model_suffix}_model.pkl"
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
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
            # Drop only role dummies (constants within the subset)
            df_subset.drop(columns=[c for c in df_subset.columns if c.startswith('role_')], inplace=True)
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
            df_subset.drop(columns=[c for c in df_subset.columns if c.startswith('role_')], inplace=True)
            train_and_save_model(df_subset, 'target_esMeta', role_name, 'esMeta')

    print("\n🎉 All models trained successfully.")


if __name__ == "__main__":
    main()
