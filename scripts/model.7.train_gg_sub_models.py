# model.7.train_gg_sub_models.py

import pandas as pd
import numpy as np
import joblib
import sys
import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass
from pathlib import Path

# --- Dependency Check ---
try:
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import GroupShuffleSplit
except ImportError:
    print("❌ Critical Error: 'scikit-learn' is not installed.")
    print("👉 Run: pip install scikit-learn")
    sys.exit(1)

# --- Configuration ---
BASE_DATA_DIR = Path(__file__).resolve().parent / '../data'
MODEL_DIR = Path(__file__).resolve().parent / '../models'
DATA_FILE = BASE_DATA_DIR / 'training_dataset_gg_sub_abs.csv'  # built by step 6

def _drop_constant_onehots(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = []
    for c in df.columns:
        if c.startswith(("role_", "bodytype_", "accelerateType_", "foot_")):
            if df[c].nunique(dropna=False) <= 1:
                to_drop.append(c)
    return df.drop(columns=to_drop)

# Minimum DISTINCT players (not rows) to train a role model. model.6 emits ~19
# highly-correlated chem-style rows per (player, role), so a row count is a poor
# proxy for information — 80 rows was only ~4 players.
MIN_PLAYERS = int(__import__('os').environ.get('MIN_PLAYERS', 30))


def _fit_scaled(X, y):
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
    model = ElasticNetCV(
        l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.8, 1.0],
        cv=5, random_state=42, n_jobs=-1, max_iter=5000, positive=False
    ).fit(X_scaled, y_scaled)
    return model, feature_scaler, target_scaler


def _metrics(model, feature_scaler, target_scaler, X, y):
    y_pred = target_scaler.inverse_transform(model.predict(feature_scaler.transform(X)).reshape(-1, 1)).ravel()
    return r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred))


def train_and_save_model(df_subset, target_col, role_name, model_suffix):
    groups = df_subset['eaId'].astype(str) if 'eaId' in df_subset.columns else None
    n_players = groups.nunique() if groups is not None else len(df_subset)
    if n_players < MIN_PLAYERS:
        print(f"Skipping '{role_name} - {model_suffix}' (only {n_players} distinct players).")
        return

    print(f"\n--- Training: {role_name} | {model_suffix} ({n_players} players / {len(df_subset)} rows) ---")

    X = df_subset.drop(columns=[c for c in (target_col, 'eaId') if c in df_subset.columns]).copy()
    y = df_subset[target_col].astype(float)

    # Honest out-of-sample metric: hold out 20% of PLAYERS so a player's correlated
    # chem-style rows never straddle train/test (which would inflate the score).
    if groups is not None:
        try:
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
            m_cv, fs_cv, ts_cv = _fit_scaled(X.iloc[tr], y.iloc[tr])
            r2_oos, rmse_oos = _metrics(m_cv, fs_cv, ts_cv, X.iloc[te], y.iloc[te])
            print(f"  OOS (held-out players): R²={r2_oos:.4f} | RMSE={rmse_oos:.4f} "
                  f"(train {groups.iloc[tr].nunique()} / test {groups.iloc[te].nunique()} players)")
        except Exception as e:
            print(f"  ⚠️ Held-out eval skipped: {e}")

    # Production model: refit on ALL rows.
    try:
        model, feature_scaler, target_scaler = _fit_scaled(X, y)
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        return

    r2_in, rmse_in = _metrics(model, feature_scaler, target_scaler, X, y)
    nnz = int((model.coef_ != 0).sum())
    print(f"  In-sample:  R²={r2_in:.4f} | RMSE={rmse_in:.4f} | Non-zero coefs: {nnz}/{len(model.coef_)}")

    coef_unscaled = (model.coef_ / target_scaler.scale_[0]) / feature_scaler.scale_
    bundle = {
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "features": X.columns.tolist(),
        "coef_unscaled": coef_unscaled.tolist(),
    }

    safe_role_name = role_name.replace(" ", "_").replace("-", "_")
    model_filename = f"{safe_role_name}_{model_suffix}_model.pkl"
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / model_filename)
    print(f"  💾 Model saved to {model_filename}")

def main():
    print("🚀 Training ABSOLUTE ggMeta models (Step 7)...")
    MODEL_DIR.mkdir(exist_ok=True)

    if not DATA_FILE.exists():
        print(f"❌ {DATA_FILE.name} not found. Run step 6 first.")
        return

    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"❌ Error reading data: {e}")
        return

    # Train PER ROLE (role_* one-hots exist)
    role_cols = [c for c in df.columns if c.startswith('role_')]
    if not role_cols:
        print("❌ No role_* columns found. Check dataset build.")
        return

    for role_col in role_cols:
        role_name = role_col.replace('role_', '').replace('_', ' ')
        df_subset = df[df[role_col] == 1].copy()
        df_subset = _drop_constant_onehots(df_subset)

        if "target_ggMetaAbs" not in df_subset.columns:
            continue

        train_and_save_model(df_subset, target_col='target_ggMetaAbs', role_name=role_name, model_suffix='ggMetaSub')

    print("\n🎉 All ggMetaSub models trained successfully.")

if __name__ == "__main__":
    main()