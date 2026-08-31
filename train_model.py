# -*- coding: utf-8 -*-
"""
train_model.py
One-time training script for the Insurance Fraud Detection Framework.
Implements: SMOTE oversampling + XGBoost + SHAP + multi-metric evaluation
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, precision_score,
    recall_score, accuracy_score, average_precision_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "Health Insurance Fraud Claims.xlsx")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "plots"), exist_ok=True)

print("=" * 60)
print("  Insurance Fraud Detection — Model Training")
print("=" * 60)

# ── 1. Load & Preprocess ───────────────────────────────────────────────────────
print("\n[1] Loading dataset ...")
df = pd.read_excel(DATA_PATH)
print(f"    Shape: {df.shape}")

# Feature engineering
df["ClaimDate"]   = pd.to_datetime(df["ClaimDate"])
df["ClaimMonth"]     = df["ClaimDate"].dt.month
df["ClaimDayOfWeek"] = df["ClaimDate"].dt.dayofweek
df["ClaimYear"]      = df["ClaimDate"].dt.year

# Drop non-predictive identifiers
drop_cols = ["ClaimID", "PatientID", "ProviderID", "ClaimDate",
             "DiagnosisCode", "ProcedureCode", "ProviderLocation"]
df.drop(columns=drop_cols, inplace=True)

# Target
df["FraudLabel"] = (df["ClaimLegitimacy"] == "Fraud").astype(int)
df.drop(columns=["ClaimLegitimacy"], inplace=True)

print(f"    Fraud: {df['FraudLabel'].sum()} | Legitimate: {(df['FraudLabel']==0).sum()}")
print(f"    Columns after drop: {df.columns.tolist()}")

# ── 2. Train / Test Split ──────────────────────────────────────────────────────
print("\n[2] Splitting dataset (80/20 stratified) ...")
FEATURES = [c for c in df.columns if c != "FraudLabel"]
X = df[FEATURES]
y = df["FraudLabel"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"    Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 3. Preprocessor ────────────────────────────────────────────────────────────
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

print(f"\n[3] Numeric features  : {num_cols}")
print(f"    Categorical features: {cat_cols}")

# ── 4. SMOTE ───────────────────────────────────────────────────────────────────
# -- 4. SMOTE -------------------------------------------------------------------
print("\n[4] Applying SMOTE to balance training data ...")
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_enc, y_train)
print(f"    After SMOTE  -> Fraud: {y_train_sm.sum()} | Legit: {(y_train_sm==0).sum()}")

# -- 5. Train XGBoost ----------------------------------------------------------
print("\n[5] Training XGBoost classifier ...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_sm, y_train_sm, verbose=False)
print("    XGBoost training complete.")

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
print("\n[6] Evaluating on test set ...")
y_pred      = model.predict(X_test_enc)
y_prob      = model.predict_proba(X_test_enc)[:, 1]

acc         = accuracy_score(y_test, y_pred)
prec        = precision_score(y_test, y_pred)
rec         = recall_score(y_test, y_pred)
f1          = f1_score(y_test, y_pred)
roc_auc     = roc_auc_score(y_test, y_prob)
pr_auc      = average_precision_score(y_test, y_prob)

# False Positive Rate
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
gcpr    = tn / (tn + fp) if (tn + fp) > 0 else 0   # Genuine Claim Protection Rate

print(f"    Accuracy              : {acc:.4f}")
print(f"    Precision             : {prec:.4f}")
print(f"    Recall (Fraud)        : {rec:.4f}")
print(f"    F1-Score              : {f1:.4f}")
print(f"    ROC-AUC               : {roc_auc:.4f}")
print(f"    PR-AUC                : {pr_auc:.4f}")
print(f"    False Positive Rate   : {fpr_val:.4f}")
print(f"    Genuine Claim Prot.   : {gcpr:.4f}")
print(f"\n    Confusion Matrix:\n{cm}")

# Three-tier routing distribution on test set
theta1, theta2 = 0.30, 0.60
tiers = pd.cut(y_prob, bins=[-0.001, theta1, theta2, 1.001],
               labels=["Auto-Approve", "Manual Review", "Investigate"])
tier_counts = tiers.value_counts().to_dict()
print(f"\n    Risk Tier Distribution:\n    {tier_counts}")

# ── 7. Save metrics ────────────────────────────────────────────────────────────
metrics = {
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1_score": round(f1, 4),
    "roc_auc": round(roc_auc, 4),
    "pr_auc": round(pr_auc, 4),
    "fpr": round(fpr_val, 4),
    "gcpr": round(gcpr, 4),
    "confusion_matrix": cm.tolist(),
    "tier_counts": {k: int(v) for k, v in tier_counts.items()},
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "total_fraud": int(df["FraudLabel"].sum()),
    "total_legit": int((df["FraudLabel"] == 0).sum()),
    "theta1": theta1,
    "theta2": theta2,
    "feature_names": FEATURES,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ── 8. Save model & preprocessor ──────────────────────────────────────────────
print("\n[7] Saving model artifacts ...")
joblib.dump(model,        os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

# Also save feature names after OHE for SHAP
ohe_feature_names = list(preprocessor.get_feature_names_out())
with open(os.path.join(MODEL_DIR, "ohe_features.json"), "w") as f:
    json.dump(ohe_feature_names, f)

# ── 9. Generate static plots ──────────────────────────────────────────────────
print("\n[8] Generating evaluation plots ...")
PLOT_DIR = os.path.join(STATIC_DIR, "plots")

# ---- Confusion Matrix ----
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"], ax=ax)
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), dpi=120, bbox_inches="tight")
plt.close()

# ---- ROC Curve ----
fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(fpr_arr, tpr_arr, color="#6C63FF", lw=2.5, label=f"ROC-AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"), dpi=120, bbox_inches="tight")
plt.close()

# ---- PR Curve ----
prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rec_arr, prec_arr, color="#FF6584", lw=2.5, label=f"PR-AUC = {pr_auc:.3f}")
ax.axhline(y=df["FraudLabel"].mean(), color="gray", linestyle="--", lw=1, label="Baseline")
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "pr_curve.png"), dpi=120, bbox_inches="tight")
plt.close()

# ---- SHAP Global Summary ----
print("\n[9] Computing SHAP values (this may take ~30s) ...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_enc[:200])  # sample 200 for speed

fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_values, X_test_enc[:200],
                  feature_names=ohe_feature_names,
                  plot_type="bar", show=False)
plt.title("Global SHAP Feature Importance", fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "shap_summary.png"), dpi=120, bbox_inches="tight")
plt.close()

# ---- Feature Importance (XGBoost native) ----
importances = model.feature_importances_
feat_df     = pd.DataFrame({"feature": ohe_feature_names, "importance": importances})
feat_df     = feat_df.sort_values("importance", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(feat_df["feature"][::-1], feat_df["importance"][::-1], color="#6C63FF")
ax.set_xlabel("Feature Importance", fontsize=11)
ax.set_title("Top 15 Feature Importances (XGBoost)", fontsize=13, fontweight="bold")
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"), dpi=120, bbox_inches="tight")
plt.close()

print("\nAll plots saved to static/plots/")
print("Model artifacts saved to model/")
print("\n" + "=" * 60)
print("  Training Complete! Run  python app.py  to start the app.")
print("=" * 60)
