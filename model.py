# model.py
# Requisitos: numpy, pandas, scikit-learn, catboost, shap, lime, matplotlib, optuna (opcional), joblib
# pip install numpy pandas scikit-learn catboost shap lime matplotlib optuna joblib

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import List, Tuple, Any

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from lime.lime_tabular import LimeTabularExplainer
from catboost import CatBoostClassifier, Pool
import shap
import joblib

warnings.filterwarnings("ignore")
plt.rcParams["figure.autolayout"] = True

# -------------------------
# Config
# -------------------------
DATA_PATH = "data/train.csv"      # Ajusta si hace falta
TARGET_COL = "target_variable"    # Ajusta a tu target
ID_COL = "id"                     # Ajusta si procede
MEDIA_DIR = "media"
MODEL_DIR = "model"
OUTPUT_DIR = "data"
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# Utilities
# -------------------------
def save_figure_safely(filename: str, fig: plt.Figure = None) -> bool:
    """Salva una figura matplotlib si contiene ejes con contenido.
    Devuelve True si se guardó correctamente."""
    if fig is None:
        fig = plt.gcf()

    # No hay ejes -> nada que guardar
    if len(fig.axes) == 0:
        print(f"[SKIP] No content to save: {filename}")
        plt.close(fig)
        return False

    try:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        file_size = Path(filename).stat().st_size
        if file_size < 1000:  # heurística: menos de 1KB probablemente vacío/corrupto
            Path(filename).unlink(missing_ok=True)
            print(f"[WARN] Empty file removed: {filename}")
            plt.close(fig)
            return False
        print(f"[OK] Saved: {filename}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save {filename}: {e}")
        return False
    finally:
        plt.close(fig)


def ensure_numpy_floats(arr: Any) -> np.ndarray:
    """Convierte a np.ndarray de tipo float si es posible."""
    a = np.array(arr)
    try:
        return a.astype(float)
    except Exception:
        return a


# -------------------------
# Load & prepare data
# -------------------------
def load_data(path: str, target_col: str, id_col: str):
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    X = df.drop(columns=[target_col, id_col]) if id_col in df.columns else df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# -------------------------
# CatBoost CV & final train
# -------------------------
def run_catboost_cv_and_train(X: pd.DataFrame, y: pd.Series) -> Tuple[CatBoostClassifier, dict]:
    print("\n" + "=" * 60)
    print("CatBoost Classifier with Cross-Validation")
    print("=" * 60 + "\n")

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        border_count=128,
        random_seed=RANDOM_STATE,
        verbose=False,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    f1_scores, roc_auc_scores, precision_scores, recall_scores = [], [], [], []

    print("Performing 5-fold cross-validation...\n")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, y_pred)
        roc = roc_auc_score(y_val, y_proba)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)

        f1_scores.append(f1)
        roc_auc_scores.append(roc)
        precision_scores.append(prec)
        recall_scores.append(rec)

        print(f"Fold {fold}: F1={f1:.4f}, ROC-AUC={roc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

    print("\n" + "-" * 60)
    print("Cross-Validation Results (Mean ± Std):")
    print("-" * 60)
    print(f"F1 Score:     {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"ROC-AUC:      {np.mean(roc_auc_scores):.4f} ± {np.std(roc_auc_scores):.4f}")
    print(f"Precision:    {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall:       {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")

    # Entrenar modelo final con todo el training set
    model.fit(X, y)
    metrics = {
        "cv_f1_mean": float(np.mean(f1_scores)),
        "cv_f1_std": float(np.std(f1_scores)),
        "cv_roc_mean": float(np.mean(roc_auc_scores)),
        "cv_roc_std": float(np.std(roc_auc_scores)),
    }
    return model, metrics


# -------------------------
# SHAP explainability
# -------------------------
def compute_and_save_shap(cat_model: CatBoostClassifier, X_test: pd.DataFrame) -> Tuple[np.ndarray, float, List[str]]:
    explainer = shap.TreeExplainer(cat_model)
    shap_values_raw = explainer.shap_values(X_test)

    # For binary classifiers shap_values may come como lista [class0, class1]
    if isinstance(shap_values_raw, list) and len(shap_values_raw) >= 2:
        shap_vals = shap_values_raw[1]
    else:
        shap_vals = shap_values_raw

    # Normalizar dimensiones
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(-1, 1)

    print(f"\nSHAP Debug: X_test has {X_test.shape[1]} features, SHAP has {shap_vals.shape[1]} values")
    if shap_vals.shape[1] != X_test.shape[1]:
        print("[WARN] Shape mismatch: intentaré continuar con lo disponible.")

    # Summary plot (global)
    try:
        shap.summary_plot(shap_vals, X_test, show=False)
        save_figure_safely(os.path.join(MEDIA_DIR, "shap_summary.png"))
    except Exception as e:
        print(f"[WARN] No se pudo generar shap_summary: {e}")

    # Bar plot
    try:
        fig = plt.figure()
        shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False)
        save_figure_safely(os.path.join(MEDIA_DIR, "shap_bar.png"), fig)
    except Exception as e:
        print(f"[WARN] No se pudo generar shap_bar: {e}")

    # Force plot for first instance (matplotlib)
    try:
        i = 0
        shap.initjs()
        expected_val = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
        plt.figure()
        shap.force_plot(expected_val, shap_vals[i, :], X_test.iloc[i, :], matplotlib=True, show=False)
        save_figure_safely(os.path.join(MEDIA_DIR, "shap_force_example.png"))
    except Exception as e:
        print(f"[WARN] No se pudo generar shap_force_example: {e}")

    # Top features by mean abs SHAP
    importances = np.abs(shap_vals).mean(axis=0)
    # If dimensional mismatch, clip to available columns
    n_feats_shap = shap_vals.shape[1]
    columns_for_shap = list(X_test.columns)[:n_feats_shap]
    top_idx = np.argsort(importances)[::-1][:min(10, len(importances))]
    top_features = [columns_for_shap[i] for i in top_idx if i < len(columns_for_shap)]

    return shap_vals, expected_val if 'expected_val' in locals() else 0.0, top_features


# -------------------------
# PDP usando CatBoost (solución 1)
# -------------------------
def pdp_catboost_and_save(cat_model: CatBoostClassifier, X_ref: pd.DataFrame, top_features: List[str]):
    """
    Calcula PDP manualmente para cada feature:
    - Se crea un grid de valores entre percentil 5 y 95
    - Para cada punto del grid se fuerza la columna a ese valor
    - Se calcula la media de predict_proba
    """
    print("\nGenerating PDP plots (manual PDP method)...")

    for feat in top_features:
        if feat not in X_ref.columns:
            print(f"[SKIP] Feature {feat} not in X_ref columns")
            continue

        values = X_ref[feat]
        vmin, vmax = np.percentile(values, [5, 95])
        grid = np.linspace(vmin, vmax, 30)

        pdp_y = []

        for val in grid:
            X_temp = X_ref.copy()
            X_temp[feat] = val
            proba = cat_model.predict_proba(X_temp)[:, 1]
            pdp_y.append(proba.mean())

        # Plot
        plt.figure(figsize=(6, 4))
        plt.plot(grid, pdp_y)
        plt.xlabel(feat)
        plt.ylabel("Partial Dependence")
        plt.title(f"PDP - {feat}")

        out_file = os.path.join(MEDIA_DIR, f"pdp_{feat}.png")
        save_figure_safely(out_file)

# -------------------------
# Save predictions + SHAP to CSV/JSON
# -------------------------
def save_predictions_with_shap(X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray, shap_vals: np.ndarray, expected_val: float):
    preds_df = X_test.copy().reset_index(drop=True)
    preds_df["prediction"] = y_pred
    preds_df["prediction_probability"] = y_proba
    preds_df["actual_outcome"] = y_test.values
    preds_df["correct_prediction"] = (y_pred == y_test.values).astype(int)

    # Add SHAP columns (safe with possible mismatches)
    n_shap = shap_vals.shape[1]
    n_feats = len(X_test.columns)
    for idx in range(min(n_shap, n_feats)):
        col = X_test.columns[idx]
        preds_df[f"shap_{col}"] = shap_vals[:, idx]

    preds_df["shap_base_value"] = expected_val
    preds_df["shap_sum"] = shap_vals.sum(axis=1)

    out_csv = os.path.join(OUTPUT_DIR, "predictions_with_shap.csv")
    preds_df.to_csv(out_csv, index=False)
    print(f"[OK] Saved predictions with SHAP values to: {out_csv}")

    # JSON detallado
    detailed = []
    for idx in range(len(preds_df)):
        n_use = min(shap_vals.shape[1], len(X_test.columns))
        feature_values = {X_test.columns[i]: float(X_test.iloc[idx, i]) for i in range(n_use)}
        shap_values_dict = {X_test.columns[i]: float(shap_vals[idx, i]) for i in range(n_use)}
        sorted_features = sorted(shap_values_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        d = {
            "test_index": int(idx),
            "prediction": int(y_pred[idx]),
            "prediction_probability": float(y_proba[idx]),
            "actual_outcome": int(y_test.iloc[idx]),
            "correct_prediction": bool(y_pred[idx] == int(y_test.iloc[idx])),
            "feature_values": feature_values,
            "shap_values": shap_values_dict,
            "shap_base_value": float(expected_val),
            "top_positive_features": [(feat, val) for feat, val in sorted_features if val > 0][:5],
            "top_negative_features": [(feat, val) for feat, val in sorted_features if val < 0][:5],
            "top_absolute_features": sorted_features[:5],
        }
        detailed.append(d)

    json_file = os.path.join(OUTPUT_DIR, "predictions_detailed.json")
    with open(json_file, "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"[OK] Saved detailed predictions to: {json_file}")


# -------------------------
# Main
# -------------------------
def main():
    X, y = load_data(DATA_PATH, TARGET_COL, ID_COL)

    # Split train/test stratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 1) CV + train final (entrena con X_train para CV y después con todo X_train? aquí entrenamos final sobre X_train)
    # Nota: la función run_catboost_cv_and_train entrena sobre lo que le pasamos; para que coincida con tu pipeline
    # podemos 1) ejecutar CV sobre X_train (ya hecho) y luego entrenar final con X_train (o con X if prefieres)
    cat_model, cv_metrics = run_catboost_cv_and_train(X_train, y_train)

    # Entrenar final sobre todo el set de entrenamiento original (X_train). 
    # Si quieres entrenar sobre X_train + X_test, cambia aquí.
    cat_model.fit(X_train, y_train)

    # Guardar modelo
    model_path = os.path.join(MODEL_DIR, "catboost_model.joblib")
    joblib.dump(cat_model, model_path)
    print(f"[OK] Model saved to: {model_path}")

    # Evaluación en test
    y_pred = cat_model.predict(X_test)
    y_proba = cat_model.predict_proba(X_test)[:, 1]

    print("\n" + "-" * 60)
    print("Final CatBoost Model - Test Set Evaluation:")
    print("-" * 60)
    print(f"F1 Score:     {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:      {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Precision:    {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:       {recall_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 60 + "\n")

    # SHAP
    shap_vals, expected_val, top_features_shap = compute_and_save_shap(cat_model, X_test)

    # PDP: usa CatBoost native method para las top-3 features (o top_features_shap[:3])
    top_for_pdp = top_features_shap[:3] if len(top_features_shap) >= 1 else list(X_test.columns[:3])
    pdp_catboost_and_save(cat_model, X_test.copy(), top_for_pdp)

    # Guardar predicciones y SHAP
    save_predictions_with_shap(X_test, y_test, y_pred, y_proba, shap_vals, expected_val)

    # Resumen
    print("\n" + "-" * 60)
    print("Prediction Summary:")
    print("-" * 60)
    total = len(y_pred)
    won = int((y_pred == 1).sum())
    lost = total - won
    correct = int((y_pred == y_test.values).sum())
    print(f"Total predictions: {total}")
    print(f"Predicted WON: {won} ({won/total*100:.1f}%)")
    print(f"Predicted LOST: {lost} ({lost/total*100:.1f}%)")
    print(f"Correct predictions: {correct} ({correct/total*100:.1f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
