# AI Governance Framework for Concept Drift Detection

A dataset-agnostic AI governance framework that monitors machine learning models
for concept drift and ensures reliable decision-making in production environments.

---

## Team

| Name | USN |
|---|---|
| M Saranya | R23EO013 |
| Suchitra S | R23EQ116 |
| Syed Umer S | R23EQ121 |
| Vishal Thomas | R23EQ133 |

**Guide:** Dr. B. MuthuKumar  
**Institution:** REVA University, Dept. of Computer Science & Engineering

---

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000 in your browser.

### Quick demo (no browser needed)

```bash
python run_demo.py
```

---

## How It Works

1. **Upload** any CSV or Excel dataset, or choose a bundled sample
2. **Auto-detection** — problem type (binary/multiclass classification or regression) is inferred from the target column
3. **Model training** — 9–12 individual and ensemble models are trained and compared automatically
4. **Drift simulation** — the dataset is split into a baseline (60%) and an incoming stream (40%); a synthetic drifted batch is also generated with Gaussian noise
5. **Drift detection** — PSI, KL Divergence, and ADWIN are run on every incoming batch
6. **Alerts** — critical/warning/info alerts are generated automatically based on thresholds
7. **Report** — download a self-contained HTML report with charts, tables, and recommendations

---

## Project Structure

```
├── app.py                    Flask application & all API routes
├── run_demo.py               Standalone CLI demo
├── requirements.txt
├── README.md
│
├── data/
│   ├── dataset_handler.py    DatasetHandler — auto-detect, preprocess, split
│   ├── file_upload.py        FileUploadHandler — validate & save uploads
│   ├── generate_samples.py   Generates 3 sample CSVs on first run
│   └── samples/
│       ├── credit_fraud.csv
│       ├── employee_churn.csv
│       └── house_prices.csv
│
├── models/
│   ├── train_models.py       ModelFactory — adaptive model catalogue + comparison
│   └── saved/                Persisted model artefacts (future use)
│
├── drift/
│   ├── psi_detector.py       PSIDetector — numeric & categorical PSI
│   └── drift_manager.py      DriftManager — baseline storage, KL, explanations
│
├── reports/
│   └── report_generator.py   ReportGenerator — self-contained HTML reports
│
├── static/
│   ├── css/style.css         Dark theme
│   └── js/dashboard.js       Shared JS — charts, KPI cards, auto-refresh
│
├── templates/
│   ├── _navbar.html          Shared navbar partial
│   ├── upload.html           Landing / upload page
│   ├── dashboard.html        Main dashboard
│   ├── model_comparison.html Model comparison page
│   ├── drift_monitor.html    Drift monitoring page
│   └── alerts.html           Alert center
│
└── logs/
    └── app.log               Application logs
```

---

## Drift Detection Methods

### PSI (Population Stability Index)
Compares feature distributions between baseline and incoming data.
- Numeric columns: bucket-based PSI using equal-width bins from baseline
- Categorical columns: frequency-proportion PSI across all categories
- Thresholds: < 0.1 No Drift | 0.1–0.2 Slight | 0.2–0.25 Moderate | > 0.25 Severe

### KL Divergence
Measures the information-theoretic distance between two probability distributions.
Applied to numeric features using shared histogram bins.
Threshold: 0.1

### ADWIN (Adaptive Windowing)
Online drift detection via the River library. Maintains an adaptive window over
the prediction correctness stream and signals drift when the window mean changes
significantly. No fixed threshold required.

---

## Model Comparison

The framework tests models in two groups:

**Individual models**
- Logistic Regression, Random Forest, Decision Tree, Gradient Boosting
- SVM (skipped if n_rows ≥ 10,000), KNN (skipped if n_cols ≥ 20), Naive Bayes

**Ensemble / combination models**
- VotingClassifier (hard & soft), StackingClassifier, BaggingClassifier, AdaBoost

Selection criterion: highest F1-score (classification) or R² (regression).

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/upload` | Upload page |
| GET | `/dashboard` | Main dashboard |
| POST | `/api/upload-dataset` | Upload & analyze a dataset |
| POST | `/api/set-target-column` | Override auto-detected target |
| POST | `/api/run-pipeline` | Run full pipeline |
| GET | `/api/datasets` | List all datasets |
| DELETE | `/api/datasets/<id>` | Delete a dataset |
| GET/POST | `/api/active-dataset` | Get or set active dataset |
| GET | `/api/performance-metrics` | Model performance data |
| GET | `/api/model-comparison` | Full model comparison data |
| GET | `/api/drift-status` | Drift detection results |
| POST | `/api/simulate-batch` | Simulate next incoming batch |
| POST | `/api/retrain` | Retrain best model |
| GET | `/api/generate-report` | Download HTML report |
| GET | `/api/alerts` | List all alerts |
| POST | `/api/alerts/<id>/read` | Mark alert as read |

---

## Sample Datasets

| Dataset | Rows | Type | Target |
|---|---|---|---|
| credit_fraud.csv | 2,000 | Binary Classification | fraud |
| employee_churn.csv | 1,500 | Binary Classification | churn |
| house_prices.csv | 1,000 | Regression | price |

---

## Requirements

See `requirements.txt`. Key dependencies:
- Flask, scikit-learn, pandas, numpy
- matplotlib (report charts)
- scipy (KL divergence)
- Werkzeug (file upload)
