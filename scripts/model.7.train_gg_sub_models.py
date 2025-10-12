# model.7b.train_gg_sub_models.py
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
DATA_FILE = BASE_DATA_DIR / 'training_dataset_gg_sub.csv'

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c.startswith("attribute") or c in ("height","weight","skillMoves","weakFoot","familiarity"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def train_and_save_model(df_subset, target_col, role_name, model_suffix):
    if len(df_subset) < 50:
        print(f"Skipping '{role_name} - {model_suffix}' (only {len(df_subset)} samples).")
        return

    print(f"\n--- Training: {role_name} | {model_suffix} ({len(df_subset)} samples) ---")
    X = df_subset.drop(columns=[target_col]).copy()
    y = df_subset[target_col].astype(float)

    # numeric guard
    X = _clean_columns(X).fillna(0)
    non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        print(f"  ⚠️ Dropping non-numeric columns: {non_num[:6]}{'...' if len(non_num)>6 else ''}")
        X = X.drop(columns=non_num)

    # scale
    feature_scaler = StandardScaler()
    X_scaled = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)

    target_scaler = MinMaxScaler((0,1))
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1,1)).ravel()

    # non-negative linear model
    model = ElasticNetCV(
        l1_ratio=[0.05,0.1,0.3,0.5,0.8,1.0],
        cv=5, random_state=42, n_jobs=-1, max_iter=5000, positive=True
    ).fit(X_scaled.values, y_scaled)

    # evaluate on train (diagnostic)
    y_pred_scaled = model.predict(X_scaled.values)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    r2 = r2_score(y, y_pred); rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"  R²: {r2:.4f} | RMSE: {rmse:.4f} | Non-zero: {(model.coef_!=0).sum()}/{len(model.coef_)}")

    # unscaled weights (for delta anchoring)
    coef_unscaled = (model.coef_ / target_scaler.scale_[0]) / feature_scaler.scale_

    bundle = {
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "features": X.columns.tolist(),
        "coef_unscaled": coef_unscaled.tolist()
    }

    safe_role = role_name.replace(" ","_").replace("-","_")
    out = MODEL_DIR / f"{safe_role}_{model_suffix}_model.pkl"
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, out)
    print(f"  💾 Saved {out.name}")

def main():
    print("🚀 Training ggMetaSub (Basic) models...")
    if not DATA_FILE.exists():
        print(f"❌ {DATA_FILE.name} not found. Run model.6b.build_gg_sub_dataset.py first.")
        return

    df = pd.read_csv(DATA_FILE)
    role_cols = [c for c in df.columns if c.startswith("role_")]

    for role_col in role_cols:
        role_name = role_col.replace("role_","").replace("_"," ")
        df_subset = df[df[role_col]==1].copy()

        # drop constant OHEs for this role subset
        to_drop = []
        for c in df_subset.columns:
            if c.startswith(("role_","bodytype_","accelerateType_","foot_")) and df_subset[c].nunique(dropna=False)<=1:
                to_drop.append(c)
        df_subset.drop(columns=to_drop, inplace=True, errors="ignore")

        train_and_save_model(df_subset, target_col="target_ggMetaSub", role_name=role_name, model_suffix="ggMetaSub")

    print("\n🎉 All ggMetaSub models trained.")

if __name__ == "__main__":
    main()
