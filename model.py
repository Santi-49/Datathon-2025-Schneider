# Required packages:
# pip install numpy pandas scikit-learn lightgbm xgboost catboost shap lime pdpbox matplotlib optuna

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
from pdpbox import pdp, info_plots
from lime.lime_tabular import LimeTabularExplainer
import optuna
import warnings

warnings.filterwarnings("ignore")

# 1) Load data (adjust path)
df = pd.read_csv("train.csv")  # replace with your file
target = "target_variable"
id_col = "id"

# 2) Basic EDA & feature list
X = df.drop(columns=[target, id_col])
y = df[target]

# If categorical features exist, encode them (example: opp_month categorical)
# For simplicity assume numeric or already encoded. If not, use pd.get_dummies or OrdinalEncoder.

# 3) Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# 4) Quick optuna tuning for LightGBM (for speed)
def objective(trial):
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
        "seed": 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    cv = lgb.cv(
        param,
        dtrain,
        nfold=4,
        stratified=True,
        callbacks=[lgb.early_stopping(stopping_rounds=20)],
        num_boost_round=2000,
    )
    # use min logloss
    return min(cv["valid binary_logloss-mean"])


# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=30, show_progress_bar=True)

# best_params = study.best_params
# best_params.update(
#     {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 42}
# )
# model = lgb.LGBMClassifier(**best_params, n_estimators=500)
# model.fit(X_train, y_train)

# 4.5) CatBoost with fixed standard parameters and CV evaluation
print("\n" + "=" * 60)
print("CatBoost Classifier with Cross-Validation")
print("=" * 60)

# Fixed standard parameters for CatBoost
catboost_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3.0,
    border_count=128,
    random_seed=42,
    verbose=False,
)

# Cross-validation with StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []
roc_auc_scores = []
precision_scores = []
recall_scores = []

print("\nPerforming 5-fold cross-validation...\n")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Train CatBoost
    catboost_model.fit(X_fold_train, y_fold_train)

    # Predictions
    y_fold_pred = catboost_model.predict(X_fold_val)
    y_fold_proba = catboost_model.predict_proba(X_fold_val)[:, 1]

    # Metrics
    f1 = f1_score(y_fold_val, y_fold_pred)
    roc_auc = roc_auc_score(y_fold_val, y_fold_proba)
    precision = precision_score(y_fold_val, y_fold_pred)
    recall = recall_score(y_fold_val, y_fold_pred)

    f1_scores.append(f1)
    roc_auc_scores.append(roc_auc)
    precision_scores.append(precision)
    recall_scores.append(recall)

    print(
        f"Fold {fold}: F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}"
    )

# Print CV results
print("\n" + "-" * 60)
print("Cross-Validation Results (Mean ± Std):")
print("-" * 60)
print(f"F1 Score:     {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"ROC-AUC:      {np.mean(roc_auc_scores):.4f} ± {np.std(roc_auc_scores):.4f}")
print(f"Precision:    {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
print(f"Recall:       {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")

# Train final CatBoost model on full training set and evaluate on test set
print("\n" + "-" * 60)
print("Final CatBoost Model - Test Set Evaluation:")
print("-" * 60)
catboost_model.fit(X_train, y_train)
y_pred_catboost = catboost_model.predict(X_test)
y_proba_catboost = catboost_model.predict_proba(X_test)[:, 1]

print(f"F1 Score:     {f1_score(y_test, y_pred_catboost):.4f}")
print(f"ROC-AUC:      {roc_auc_score(y_test, y_proba_catboost):.4f}")
print(f"Precision:    {precision_score(y_test, y_pred_catboost):.4f}")
print(f"Recall:       {recall_score(y_test, y_pred_catboost):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_catboost))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_catboost))
print("=" * 60 + "\n")

# 5) Evaluate (using CatBoost model)
# y_pred = catboost_model.predict(X_test)
# y_proba = catboost_model.predict_proba(X_test)[:, 1]
# print("F1:", f1_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print("ROC-AUC:", roc_auc_score(y_test, y_proba))
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion matrix:\n", cm)

# 6) Global explainability: SHAP
explainer = shap.TreeExplainer(catboost_model)
shap_values = explainer.shap_values(
    X_test
)  # for binary, shap_values is list: [class0, class1] maybe. Use shap_values[1]
# if it returns list:
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

# Verify SHAP values shape matches features
print(
    f"\nSHAP Debug: X_test has {X_test.shape[1]} features, SHAP has {shap_vals.shape[1]} values"
)
if shap_vals.shape[1] != X_test.shape[1]:
    print(f"WARNING: Shape mismatch detected. This may cause issues.")
    print(f"X_test columns: {list(X_test.columns)}")

# Create media folder if it doesn't exist
import os

os.makedirs("media", exist_ok=True)

# Summary plot (global)
shap.summary_plot(shap_vals, X_test, show=False)
plt.tight_layout()
plt.savefig("media/shap_summary.png", dpi=150)

# Bar plot of mean absolute SHAP (feature importance)
shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("media/shap_bar.png", dpi=150)

# 7) Local explainability: SHAP force plot for one instance
i = 0  # choose index in X_test
shap.initjs()
# For CatBoost, expected_value might be scalar or array
expected_val = (
    explainer.expected_value[1]
    if isinstance(explainer.expected_value, (list, np.ndarray))
    else explainer.expected_value
)
force_plot = shap.force_plot(
    expected_val, shap_vals[i, :], X_test.iloc[i, :], matplotlib=True
)
plt.savefig("media/shap_force_example.png", dpi=150, bbox_inches="tight")

# 8) LIME: local explanation
explainer_lime = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=["lost", "won"],
    mode="classification",
)
exp = explainer_lime.explain_instance(
    X_test.iloc[i].to_numpy(), catboost_model.predict_proba, num_features=8
)
exp.save_to_file("media/lime_example.html")

# 9) PDP on top 2 features by SHAP mean abs
importances = np.abs(shap_vals).mean(axis=0)
top_idx = np.argsort(importances)[::-1][:3]
top_features = X_test.columns[top_idx].tolist()
print("Top features by SHAP:", top_features)

# Make a copy of X_test for PDP since pdpbox modifies the dataframe
X_test_pdp = X_test.copy()

for feat in top_features:
    pdp_dist = pdp.PDPIsolate(
        model=catboost_model,
        df=X_test_pdp,
        model_features=X_test.columns.tolist(),
        feature=feat,
        feature_name=feat,
        n_classes=2,
    )
    fig, axes = pdp_dist.plot(center=True, plot_pts_dist=True)
    plt.savefig(f"media/pdp_{feat}.png", dpi=150)
    plt.close()

# 10) ALE: (use an ALE implementation if available)
# If you have 'alibi' or 'pyALE', compute ALE for top features; else skip or approximate with PDP.

# Check X_test shape after PDP (PDP might modify it)
print(f"\nX_test shape after PDP: {X_test.shape}")
print(f"X_test columns: {list(X_test.columns)}")

# 11) Save model & artifacts
import joblib
import json

os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)
joblib.dump(catboost_model, "model/catboost_model.joblib")
X_test.to_csv("data/X_test.csv", index=False)

# 12) Generate comprehensive predictions output with SHAP values
print("\n" + "=" * 60)
print("Generating Predictions Output with SHAP Explainability")
print("=" * 60)

# Debug: Check shapes
print(f"X_test shape: {X_test.shape}")
print(f"shap_vals shape: {shap_vals.shape}")
print(f"Number of features in X_test: {len(X_test.columns)}")

# Create a comprehensive dataframe with predictions and explanations
predictions_df = X_test.copy()
predictions_df["prediction"] = y_pred_catboost
predictions_df["prediction_probability"] = y_proba_catboost
predictions_df["actual_outcome"] = y_test.values
predictions_df["correct_prediction"] = (y_pred_catboost == y_test.values).astype(int)

# Add SHAP values for each feature
# Make sure the shapes match
if shap_vals.shape[1] == len(X_test.columns):
    for idx, col in enumerate(X_test.columns):
        predictions_df[f"shap_{col}"] = shap_vals[:, idx]
else:
    print(f"WARNING: SHAP values shape mismatch!")
    print(f"Expected {len(X_test.columns)} features, got {shap_vals.shape[1]}")
    # Use only available features
    for idx in range(min(shap_vals.shape[1], len(X_test.columns))):
        col = X_test.columns[idx]
        predictions_df[f"shap_{col}"] = shap_vals[:, idx]

# Add base value (expected value)
predictions_df["shap_base_value"] = expected_val

# Calculate total SHAP contribution
predictions_df["shap_sum"] = shap_vals.sum(axis=1)

# Reset index to get original test indices
predictions_df = predictions_df.reset_index(drop=True)
predictions_df.insert(0, "test_index", range(len(predictions_df)))

# Save to CSV
output_file = "data/predictions_with_shap.csv"
predictions_df.to_csv(output_file, index=False)
print(f"\n[OK] Saved predictions with SHAP values to: {output_file}")
print(f"  Total instances: {len(predictions_df)}")
print(f"  Columns: {len(predictions_df.columns)}")

# Also create a JSON file with detailed explanations for easier processing
detailed_predictions = []

for idx in range(len(X_test)):
    # Get feature values - but only for features that have SHAP values
    # Use only the first N features that match SHAP dimensions
    n_shap_features = shap_vals.shape[1]
    feature_values_dict = {}
    shap_values_dict = {}

    for i in range(n_shap_features):
        col = X_test.columns[i]
        feature_values_dict[col] = float(
            X_test.iloc[idx, i]
        )  # Convert to float for JSON
        shap_values_dict[col] = float(shap_vals[idx, i])

    # Sort features by absolute SHAP value
    sorted_features = sorted(
        shap_values_dict.items(), key=lambda x: abs(x[1]), reverse=True
    )

    instance_data = {
        "test_index": idx,
        "prediction": int(y_pred_catboost[idx]),
        "prediction_label": "WON" if y_pred_catboost[idx] == 1 else "LOST",
        "prediction_probability": float(y_proba_catboost[idx]),
        "actual_outcome": int(y_test.iloc[idx]),
        "actual_label": "WON" if y_test.iloc[idx] == 1 else "LOST",
        "correct_prediction": bool(y_pred_catboost[idx] == y_test.iloc[idx]),
        "feature_values": feature_values_dict,
        "shap_values": shap_values_dict,
        "shap_base_value": float(expected_val) if expected_val is not None else 0.0,
        "top_positive_features": [
            (feat, float(val)) for feat, val in sorted_features if val > 0
        ][:5],
        "top_negative_features": [
            (feat, float(val)) for feat, val in sorted_features if val < 0
        ][:5],
        "top_absolute_features": [(feat, float(val)) for feat, val in sorted_features][
            :5
        ],
    }

    detailed_predictions.append(instance_data)

# Save detailed JSON
json_file = "predictions_detailed.json"
with open(json_file, "w") as f:
    json.dump(detailed_predictions, f, indent=2)

print(f"[OK] Saved detailed predictions to: {json_file}")

# Print summary statistics
print("\n" + "-" * 60)
print("Prediction Summary:")
print("-" * 60)
print(f"Total predictions: {len(predictions_df)}")
print(
    f"Predicted WON: {(y_pred_catboost == 1).sum()} ({(y_pred_catboost == 1).sum() / len(y_pred_catboost) * 100:.1f}%)"
)
print(
    f"Predicted LOST: {(y_pred_catboost == 0).sum()} ({(y_pred_catboost == 0).sum() / len(y_pred_catboost) * 100:.1f}%)"
)
print(
    f"Correct predictions: {(y_pred_catboost == y_test.values).sum()} ({(y_pred_catboost == y_test.values).sum() / len(y_pred_catboost) * 100:.1f}%)"
)
print("=" * 60 + "\n")
