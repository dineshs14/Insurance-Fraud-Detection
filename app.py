# -*- coding: utf-8 -*-
"""
app.py  -- Flask backend for Insurance Fraud Detection Framework
"""

import os, json, io, base64, warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from flask import Flask, render_template, request, jsonify, send_file
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "model")
DATA_PATH  = os.path.join(BASE_DIR, "Health Insurance Fraud Claims.xlsx")

app = Flask(__name__)

# ── Load artifacts ──────────────────────────────────────────────────────────────
model        = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))

with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

with open(os.path.join(MODEL_DIR, "ohe_features.json")) as f:
    OHE_FEATURES = json.load(f)

explainer = shap.TreeExplainer(model)

THETA1 = METRICS["theta1"]
THETA2 = METRICS["theta2"]

# ── Dataset for analytics ───────────────────────────────────────────────────────
raw_df = pd.read_excel(DATA_PATH)
raw_df["ClaimDate"] = pd.to_datetime(raw_df["ClaimDate"])

# ── Helper functions ────────────────────────────────────────────────────────────
def score_to_tier(prob):
    if prob < THETA1:
        return {"tier": "Auto-Approve", "color": "success", "icon": "✅",
                "action": "Claim auto-approved. Low fraud risk detected.",
                "badge": "LOW RISK"}
    elif prob < THETA2:
        return {"tier": "Manual Review", "color": "warning", "icon": "⚠️",
                "action": "Claim flagged for fast-track manual verification.",
                "badge": "MEDIUM RISK"}
    else:
        return {"tier": "Investigate", "color": "danger", "icon": "🚨",
                "action": "Escalate immediately to Fraud Investigation Unit (FIU).",
                "badge": "HIGH RISK"}


def preprocess_single(form_data):
    """Convert form inputs to model-ready array."""
    df_row = pd.DataFrame([{
        "ClaimAmount":            float(form_data["ClaimAmount"]),
        "PatientAge":             int(form_data["PatientAge"]),
        "PatientIncome":          float(form_data["PatientIncome"]),
        "Cluster":                int(form_data["Cluster"]),
        "ClaimMonth":             int(form_data["ClaimMonth"]),
        "ClaimDayOfWeek":         int(form_data["ClaimDayOfWeek"]),
        "ClaimYear":              int(form_data.get("ClaimYear", 2024)),
        "PatientGender":          form_data["PatientGender"],
        "ProviderSpecialty":      form_data["ProviderSpecialty"],
        "ClaimStatus":            form_data["ClaimStatus"],
        "PatientMaritalStatus":   form_data["PatientMaritalStatus"],
        "PatientEmploymentStatus":form_data["PatientEmploymentStatus"],
        "ClaimType":              form_data["ClaimType"],
        "ClaimSubmissionMethod":  form_data["ClaimSubmissionMethod"],
    }])
    return preprocessor.transform(df_row)


def generate_shap_plot(X_enc, claim_id="Claim"):
    """Return base64 SHAP waterfall image for a single claim."""
    sv = explainer.shap_values(X_enc)
    sv_flat = sv[0] if sv.ndim > 1 else sv

    feat_names = OHE_FEATURES
    pairs = sorted(zip(feat_names, sv_flat), key=lambda x: abs(x[1]), reverse=True)[:12]
    names, vals = zip(*pairs)

    colors = ["#FF6584" if v > 0 else "#6C63FF" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(names)), vals, color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("SHAP Value (impact on fraud probability)", fontsize=9)
    ax.set_title(f"SHAP Explanation — {claim_id}", fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")
    red_patch   = mpatches.Patch(color="#FF6584", label="Increases Fraud Risk")
    blue_patch  = mpatches.Patch(color="#6C63FF", label="Decreases Fraud Risk")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc="lower right")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Routes ──────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    # Compute real-time KPIs from raw data
    total        = len(raw_df)
    fraud_count  = (raw_df["ClaimLegitimacy"] == "Fraud").sum()
    legit_count  = total - fraud_count
    fraud_pct    = round(fraud_count / total * 100, 1)

    # Monthly trend
    monthly = raw_df.groupby([raw_df["ClaimDate"].dt.to_period("M"), "ClaimLegitimacy"]).size().unstack(fill_value=0).reset_index()
    monthly["Month"] = monthly["ClaimDate"].astype(str)
    months      = monthly["Month"].tolist()
    fraud_trend = monthly.get("Fraud", pd.Series([0]*len(monthly))).tolist()
    legit_trend = monthly.get("Legitimate", pd.Series([0]*len(monthly))).tolist()

    # Claim type distribution
    type_dist = raw_df["ClaimType"].value_counts().to_dict()

    # Provider specialty fraud rate
    prov = raw_df.groupby("ProviderSpecialty")["ClaimLegitimacy"].apply(
        lambda x: (x == "Fraud").sum() / len(x) * 100).round(1).to_dict()

    # Amount distribution buckets
    bins  = [0, 2000, 4000, 6000, 8000, 10000]
    labels = ["<2k", "2k-4k", "4k-6k", "6k-8k", "8k+"]
    raw_df["AmtBin"] = pd.cut(raw_df["ClaimAmount"], bins=bins, labels=labels)
    amt_dist = raw_df.groupby("AmtBin", observed=True)["ClaimLegitimacy"].apply(
        lambda x: (x=="Fraud").sum()).to_dict()

    metrics = METRICS.copy()
    return render_template("index.html",
        total=total, fraud_count=fraud_count, legit_count=legit_count,
        fraud_pct=fraud_pct, metrics=metrics,
        months=json.dumps(months),
        fraud_trend=json.dumps(fraud_trend),
        legit_trend=json.dumps(legit_trend),
        type_dist=json.dumps(type_dist),
        prov_dist=json.dumps(prov),
        amt_dist=json.dumps(amt_dist),
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            X_enc = preprocess_single(request.form)
            prob  = float(model.predict_proba(X_enc)[0, 1])
            tier  = score_to_tier(prob)
            shap_img = generate_shap_plot(X_enc, claim_id=request.form.get("ClaimID", "Claim"))

            result = {
                "prob": round(prob * 100, 1),
                "tier": tier,
                "shap_img": shap_img,
                "claim_id": request.form.get("ClaimID", "N/A"),
            }
        except Exception as e:
            result = {"error": str(e)}

    options = {
        "genders":      ["F", "M"],
        "specialties":  ["Cardiology", "Orthopedics", "Pediatrics", "Neurology", "General Practice"],
        "statuses":     ["Approved", "Denied", "Pending"],
        "marital":      ["Single", "Married", "Divorced", "Widowed"],
        "employment":   ["Employed", "Unemployed", "Retired", "Student"],
        "claim_types":  ["Inpatient", "Outpatient", "Emergency", "Routine"],
        "sub_methods":  ["Online", "Paper", "Phone"],
    }
    return render_template("predict.html", result=result, options=options)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    results = None
    error   = None
    if request.method == "POST":
        try:
            file = request.files.get("file")
            if not file:
                raise ValueError("No file uploaded.")

            ext = os.path.splitext(file.filename)[1].lower()
            if ext == ".csv":
                df_in = pd.read_csv(file)
            elif ext in [".xlsx", ".xls"]:
                df_in = pd.read_excel(file)
            else:
                raise ValueError("Unsupported file type. Please upload CSV or XLSX.")

            required_cols = ["ClaimAmount", "PatientAge", "PatientIncome", "Cluster",
                             "ClaimMonth", "ClaimDayOfWeek", "PatientGender",
                             "ProviderSpecialty", "ClaimStatus", "PatientMaritalStatus",
                             "PatientEmploymentStatus", "ClaimType", "ClaimSubmissionMethod"]
            missing = [c for c in required_cols if c not in df_in.columns]
            if missing:
                raise ValueError(f"Missing columns: {', '.join(missing)}")

            df_in["ClaimYear"] = df_in.get("ClaimYear", 2024)
            X_enc = preprocessor.transform(df_in[required_cols + ["ClaimYear"]].fillna(0))
            probs = model.predict_proba(X_enc)[:, 1]
            tiers = [score_to_tier(p)["tier"] for p in probs]
            badges = [score_to_tier(p)["badge"] for p in probs]

            df_in["FraudProbability%"] = (probs * 100).round(1)
            df_in["RiskTier"]          = tiers
            df_in["RiskBadge"]         = badges

            results = df_in.to_dict(orient="records")
            tier_summary = pd.Series(tiers).value_counts().to_dict()

        except Exception as e:
            error = str(e)

    return render_template("batch.html", results=results, error=error,
                           tier_summary=getattr(locals().get('tier_summary'), 'to_dict', lambda: locals().get('tier_summary', {}))() if results else {})


@app.route("/analytics")
def analytics():
    return render_template("analytics.html", metrics=METRICS)


@app.route("/demo")
def demo():
    return render_template("demo.html", metrics=METRICS)


# -- API Endpoints -----------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data  = request.get_json()
    X_enc = preprocess_single(data)
    prob  = float(model.predict_proba(X_enc)[0, 1])
    tier  = score_to_tier(prob)
    return jsonify({"fraud_probability": round(prob, 4), "risk_tier": tier["tier"],
                    "action": tier["action"]})


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.route("/api/dataset-stats")
def api_dataset_stats():
    specialty_fraud = raw_df.groupby("ProviderSpecialty").apply(
        lambda x: {"total": len(x), "fraud": int((x["ClaimLegitimacy"] == "Fraud").sum())}
    ).to_dict()
    return jsonify({
        "total_claims": len(raw_df),
        "fraud_claims": int((raw_df["ClaimLegitimacy"] == "Fraud").sum()),
        "avg_claim_amount": round(raw_df["ClaimAmount"].mean(), 2),
        "specialty_stats": specialty_fraud,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\nInsurance Fraud Detection App")
    print(f"   http://127.0.0.1:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
