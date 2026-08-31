# 🛡️ Insurance Fraud Detection Framework

> **AI-Powered Insurance Claim Fraud Detection** using XGBoost, SMOTE, and SHAP Explainability

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-FF6600)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-blueviolet)](https://shap.readthedocs.io/)
[![Deploy to Render](https://img.shields.io/badge/Render-Deploy-46E3B7?logo=render)](https://render.com)

---

## 📌 Overview

An end-to-end **Insurance Fraud Detection System** that uses machine learning to classify health insurance claims as *Legitimate* or *Fraudulent*. The framework features:

- 🎯 **XGBoost Classifier** with SMOTE oversampling for imbalanced data
- 🧠 **SHAP Explainability** — per-claim waterfall explanations
- 📊 **Three-Tier Risk Routing** — Auto-Approve / Manual Review / Investigate
- 📈 **Interactive Dashboard** with real-time analytics
- 📁 **Batch Processing** — upload CSV/Excel for bulk fraud screening

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Input Claim │ ──▶ │  Preprocessor    │ ──▶ │  XGBoost Model  │
│  (Web Form)  │     │  (Scale + OHE)   │     │  (SMOTE-trained)│
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                          ┌────────────────────────────┤
                          ▼                            ▼
                   ┌──────────────┐          ┌──────────────────┐
                   │ Risk Tier    │          │  SHAP Explainer  │
                   │ Routing      │          │  (Waterfall Plot)│
                   └──────────────┘          └──────────────────┘
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/dineshs14/Insurance-Fraud-Detection.git
cd Insurance-Fraud-Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

### 4. (Optional) Retrain the Model

```bash
python train_model.py
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 94%+ |
| ROC-AUC | 0.98+ |
| PR-AUC | 0.97+ |
| Recall (Fraud) | 95%+ |

## 🖥️ Screenshots

### Dashboard
- Real-time fraud vs. legitimate claim analytics
- Monthly trend analysis with Chart.js visualizations
- Provider specialty fraud rate breakdown

### Predict
- Single claim fraud prediction with confidence score
- Three-tier risk routing (Auto-Approve / Manual Review / Investigate)
- SHAP waterfall plot explaining model decision

### Batch Processing
- Upload CSV/Excel files for bulk screening
- Downloadable results with risk tier assignments

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask, Python |
| ML Model | XGBoost, scikit-learn |
| Balancing | SMOTE (imbalanced-learn) |
| Explainability | SHAP |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Deployment | Render (Free Tier) |

## 📂 Project Structure

```
├── app.py                  # Flask web application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── Procfile                # Render deployment config
├── render.yaml             # Render blueprint
├── runtime.txt             # Python version
├── Health Insurance Fraud Claims.xlsx  # Dataset
├── model/
│   ├── model.pkl           # Trained XGBoost model
│   ├── preprocessor.pkl    # Fitted preprocessor
│   ├── metrics.json        # Evaluation metrics
│   └── ohe_features.json   # Feature names
├── templates/
│   ├── index.html          # Dashboard
│   ├── predict.html        # Single prediction
│   ├── batch.html          # Batch processing
│   ├── analytics.html      # Model analytics
│   └── demo.html           # Live demo
└── static/
    ├── css/style.css
    ├── js/main.js
    └── plots/              # Generated evaluation plots
```

## 🌐 Live Demo

🔗 **[View Live App](https://insurance-fraud-detection.onrender.com)** *(Hosted on Render — may take ~30s to wake up)*

## 👨‍💻 Author

**Dinesh S**
- GitHub: [@dineshs14](https://github.com/dineshs14)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
