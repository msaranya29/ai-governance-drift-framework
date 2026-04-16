import os
import uuid
import logging
from datetime import datetime

import pandas as pd
from flask import (
    Flask, request, jsonify, session,
    render_template, redirect, url_for,
)

from data.dataset_handler import DatasetHandler
from data.file_upload import FileUploadHandler, FileUploadError

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-change-in-prod')

UPLOAD_HANDLER = FileUploadHandler()

# In-memory registry: dataset_id → { filepath, profile, target_column, handler }
# For production swap this for a DB / Redis store.
DATASET_REGISTRY: dict = {}

# ---------------------------------------------------------------------------
# Ensure sample datasets exist on startup
# ---------------------------------------------------------------------------

def _ensure_samples():
    samples_dir = os.path.join('data', 'samples')
    sample_files = ['credit_fraud.csv', 'employee_churn.csv', 'house_prices.csv']
    if not all(os.path.exists(os.path.join(samples_dir, f)) for f in sample_files):
        logger.info("Generating sample datasets …")
        from data.generate_samples import (
            generate_credit_fraud, generate_employee_churn, generate_house_prices,
        )
        generate_credit_fraud()
        generate_employee_churn()
        generate_house_prices()

    # Register samples in the registry if not already there
    for fname in sample_files:
        fpath = os.path.join(samples_dir, fname)
        if os.path.exists(fpath) and not _find_dataset_by_path(fpath):
            _register_dataset(fpath, is_sample=True)


def _find_dataset_by_path(filepath: str):
    for did, meta in DATASET_REGISTRY.items():
        if meta['filepath'] == filepath:
            return did
    return None


def _resolve_dataset_id() -> str:
    """Return dataset_id from query param, JSON body, or session — in that order."""
    did = request.args.get('dataset_id')
    if not did and request.is_json:
        did = (request.get_json(silent=True) or {}).get('dataset_id')
    if not did:
        did = session.get('active_dataset_id')
    # Last resort: first dataset that has pipeline results
    if not did or did not in DATASET_REGISTRY:
        for k, v in DATASET_REGISTRY.items():
            if v.get('pipeline_results'):
                did = k
                break
    return did or ''


def _register_dataset(filepath: str, is_sample: bool = False) -> str:
    dataset_id = str(uuid.uuid4())
    handler = DatasetHandler(filepath)
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
    target = handler.auto_detect_target(df)
    handler.target_column = target
    profile = handler.analyze_dataset(df)

    DATASET_REGISTRY[dataset_id] = {
        'filepath':    filepath,
        'filename':    os.path.basename(filepath),
        'handler':     handler,
        'profile':     profile,
        'target_column': target,
        'is_sample':   is_sample,
        'uploaded_at': datetime.now().isoformat(),
    }
    logger.info("Registered dataset %s -> %s", dataset_id, filepath)
    return dataset_id


def _profile_to_dict(dataset_id: str) -> dict:
    meta    = DATASET_REGISTRY[dataset_id]
    profile = meta['profile']
    df      = _load_df(meta['filepath'])
    return {
        'dataset_id':         dataset_id,
        'filename':           meta['filename'],
        'n_rows':             profile.n_rows,
        'n_cols':             profile.n_columns,
        'target_column':      meta['target_column'],
        'problem_type':       profile.problem_type,
        'class_distribution': {str(k): int(v) for k, v in profile.class_distribution.items()},
        'missing_values':     {k: int(v) for k, v in profile.missing_values.items()},
        'recommended_models': profile.recommended_models,
        'feature_types':      profile.feature_types,
        'preview':            df.head(5).to_dict(orient='records'),
        'is_sample':          meta['is_sample'],
        'uploaded_at':        meta['uploaded_at'],
    }


def _load_df(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)

# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    _ensure_samples()
    active = session.get('active_dataset_id')
    if active and active in DATASET_REGISTRY:
        return redirect(url_for('dashboard'))
    return redirect(url_for('upload_page'))


@app.route('/upload')
def upload_page():
    _ensure_samples()
    return render_template('upload.html')


@app.route('/dashboard')
def dashboard():
    active = session.get('active_dataset_id')
    if not active or active not in DATASET_REGISTRY:
        return redirect(url_for('upload_page'))
    return render_template('dashboard.html')

# ---------------------------------------------------------------------------
# API — dataset management
# ---------------------------------------------------------------------------

@app.route('/api/upload-dataset', methods=['POST'])
def api_upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request.'}), 400

    file = request.files['file']
    try:
        filepath = UPLOAD_HANDLER.save(file)
    except FileUploadError as exc:
        return jsonify({'error': str(exc)}), 400

    try:
        dataset_id = _register_dataset(filepath)
    except Exception as exc:
        logger.exception("Failed to register dataset")
        return jsonify({'error': f'Could not parse dataset: {exc}'}), 422

    session['active_dataset_id'] = dataset_id
    return jsonify(_profile_to_dict(dataset_id)), 201


@app.route('/api/set-target-column', methods=['POST'])
def api_set_target_column():
    body       = request.get_json(force=True)
    dataset_id = body.get('dataset_id') or session.get('active_dataset_id')
    target_col = body.get('target_column')

    if not dataset_id or dataset_id not in DATASET_REGISTRY:
        return jsonify({'error': 'Dataset not found.'}), 404
    if not target_col:
        return jsonify({'error': 'target_column is required.'}), 400

    meta = DATASET_REGISTRY[dataset_id]
    df   = _load_df(meta['filepath'])

    if target_col not in df.columns:
        return jsonify({'error': f"Column '{target_col}' not found in dataset."}), 400

    meta['target_column']          = target_col
    meta['handler'].target_column  = target_col
    meta['profile']                = meta['handler'].analyze_dataset(df)

    return jsonify({'dataset_id': dataset_id, 'target_column': target_col,
                    'problem_type': meta['profile'].problem_type})


@app.route('/api/run-pipeline', methods=['POST'])
def api_run_pipeline():
    body       = request.get_json(force=True) or {}
    dataset_id = body.get('dataset_id') or session.get('active_dataset_id')

    if not dataset_id or dataset_id not in DATASET_REGISTRY:
        return jsonify({'error': 'Dataset not found. Upload a dataset first.'}), 404

    meta    = DATASET_REGISTRY[dataset_id]
    handler = meta['handler']
    profile = meta['profile']

    # Allow one-off target override
    target = body.get('target_column') or meta['target_column']
    session['active_dataset_id'] = dataset_id

    df = _load_df(meta['filepath'])

    try:
        # 1. Preprocess
        X, y, feature_names, scaler = handler.preprocess(df, target)

        # 2. Split with drift simulation
        X_train, X_test, y_train, y_test, X_stream, y_stream, X_drifted, y_drifted = \
            handler.split_with_drift_simulation(X, y)

        # 3. Train & compare models
        from models.train_models import ModelFactory
        factory = ModelFactory()
        models  = factory.get_models_for_problem_type(profile.problem_type, profile)
        results_df, best_model, best_name, explanation = factory.train_and_compare(
            X_train, X_test, y_train, y_test, profile.problem_type, models
        )

        # 4. Drift detection on stream vs baseline
        from drift.drift_manager import DriftManager
        baseline_df_feat = pd.DataFrame(X_train, columns=feature_names)
        stream_df_feat   = pd.DataFrame(X_stream, columns=feature_names)
        drifted_df_feat  = pd.DataFrame(X_drifted, columns=feature_names)

        dm = DriftManager(baseline_df=baseline_df_feat, feature_columns=feature_names)
        stream_drift  = dm.check_drift(stream_df_feat)
        synth_drift   = dm.check_drift(drifted_df_feat)

        # Store manager in registry for later /api/drift-status calls
        meta['drift_manager']  = dm
        meta['best_model']     = best_model
        meta['best_model_name']= best_name
        meta['feature_names']  = feature_names
        meta['scaler']         = scaler

        return jsonify({
            'dataset_id':    dataset_id,
            'problem_type':  profile.problem_type,
            'best_model':    best_name,
            'explanation':   explanation,
            'model_results': results_df.fillna('N/A').to_dict(orient='records'),
            'drift': {
                'stream': {
                    'overall_psi':      stream_drift.overall_psi,
                    'drifted_features': stream_drift.drifted_features,
                    'alert':            stream_drift.alert,
                    'explanations':     stream_drift.explanations,
                    'psi_table':        stream_drift.psi_report.to_dataframe().to_dict(orient='records'),
                },
                'synthetic': {
                    'overall_psi':      synth_drift.overall_psi,
                    'drifted_features': synth_drift.drifted_features,
                    'alert':            synth_drift.alert,
                    'explanations':     synth_drift.explanations,
                    'psi_table':        synth_drift.psi_report.to_dataframe().to_dict(orient='records'),
                },
            },
        })

    except Exception as exc:
        logger.exception("Pipeline failed for dataset %s", dataset_id)
        return jsonify({'error': str(exc)}), 500


@app.route('/api/datasets', methods=['GET'])
def api_list_datasets():
    _ensure_samples()
    return jsonify([_profile_to_dict(did) for did in DATASET_REGISTRY])


@app.route('/api/datasets/<dataset_id>', methods=['DELETE'])
def api_delete_dataset(dataset_id):
    if dataset_id not in DATASET_REGISTRY:
        return jsonify({'error': 'Dataset not found.'}), 404

    meta = DATASET_REGISTRY.pop(dataset_id)

    # Remove file only if it's a user upload (not a bundled sample)
    if not meta['is_sample']:
        try:
            os.remove(meta['filepath'])
        except OSError:
            pass

    if session.get('active_dataset_id') == dataset_id:
        session.pop('active_dataset_id', None)

    return jsonify({'deleted': dataset_id})


@app.route('/api/active-dataset', methods=['GET', 'POST'])
def api_active_dataset():
    if request.method == 'POST':
        body       = request.get_json(force=True) or {}
        dataset_id = body.get('dataset_id')
        if dataset_id not in DATASET_REGISTRY:
            return jsonify({'error': 'Dataset not found.'}), 404
        session['active_dataset_id'] = dataset_id
        return jsonify({'active_dataset_id': dataset_id})

    active = session.get('active_dataset_id')
    if not active or active not in DATASET_REGISTRY:
        return jsonify({'active_dataset_id': None})
    return jsonify(_profile_to_dict(active))


@app.route('/api/drift-status', methods=['GET'])
def api_drift_status():
    dataset_id = _resolve_dataset_id()
    if not dataset_id or dataset_id not in DATASET_REGISTRY:
        return jsonify({'error': 'No active dataset.'}), 404

    meta = DATASET_REGISTRY[dataset_id]
    dm = meta.get('drift_manager')
    if dm is None:
        return jsonify({'error': 'Pipeline not run yet for this dataset.'}), 400

    summary = dm.get_drift_summary().to_dict(orient='records')

    feature_psi = []
    if dm._history:
        latest = dm._history[-1]
        for r in latest.psi_report.results:
            feature_psi.append({
                'feature':     r.feature,
                'type':        r.feature_type,
                'psi':         round(r.psi, 4),
                'severity':    r.severity,
                'explanation': latest.explanations.get(r.feature, ''),
            })
        kl_scores   = latest.kl_scores
        overall_psi = latest.overall_psi
        drifted     = latest.drifted_features
        alert       = latest.alert
    else:
        kl_scores = {}; overall_psi = 0.0; drifted = []; alert = False

    return jsonify({
        'dataset_id':       dataset_id,
        'overall_psi':      overall_psi,
        'psi_severity':     _psi_severity_label(overall_psi),
        'drifted_features': drifted,
        'alert':            alert,
        'feature_psi':      feature_psi,
        'kl_scores':        kl_scores,
        'drift_history':    summary,
    })


@app.route('/api/performance-metrics', methods=['GET'])
def api_performance_metrics():
    dataset_id = _resolve_dataset_id()
    if not dataset_id or dataset_id not in DATASET_REGISTRY:
        return jsonify({'error': 'No active dataset.'}), 404
    meta = DATASET_REGISTRY[dataset_id]
    results = meta.get('pipeline_results')
    if not results:
        return jsonify({'error': 'Pipeline not run yet.'}), 400
    return jsonify(results)


@app.route('/api/model-comparison', methods=['GET'])
def api_model_comparison():
    dataset_id = _resolve_dataset_id()
    if not dataset_id or dataset_id not in DATASET_REGISTRY:
        return jsonify({'error': 'No active dataset.'}), 404
    meta = DATASET_REGISTRY[dataset_id]
    results = meta.get('pipeline_results')
    if not results:
        return jsonify({'error': 'Pipeline not run yet.'}), 400
    return jsonify({
        'dataset_name':  meta['filename'],
        'problem_type':  meta['profile'].problem_type,
        'best_model':    meta.get('best_model_name', ''),
        'explanation':   results.get('explanation', ''),
        'model_results': results.get('model_results', []),
    })


@app.route('/api/alerts', methods=['GET'])
def api_alerts():
    dataset_id = _resolve_dataset_id()
    alerts = _ALERTS_STORE.get(dataset_id, [])
    # Also include global alerts (no dataset_id key)
    return jsonify({'alerts': alerts, 'unread': sum(1 for a in alerts if not a.get('read'))})


@app.route('/api/alerts/<int:alert_id>/read', methods=['POST'])
def api_mark_alert_read(alert_id):
    dataset_id = _resolve_dataset_id()
    for a in _ALERTS_STORE.get(dataset_id, []):
        if a['id'] == alert_id:
            a['read'] = True
    return jsonify({'ok': True})


@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    dataset_id = _resolve_dataset_id()
    if not dataset_id or dataset_id not in DATASET_REGISTRY:
        return jsonify({'error': 'No active dataset.'}), 404

    meta    = DATASET_REGISTRY[dataset_id]
    handler = meta['handler']
    profile = meta['profile']
    target  = meta['target_column']
    df      = _load_df(meta['filepath'])

    try:
        X, y, feature_names, scaler = handler.preprocess(df, target)
        X_train, X_test, y_train, y_test, *_ = handler.split_with_drift_simulation(X, y)

        from models.train_models import ModelFactory
        factory = ModelFactory()
        models  = factory.get_models_for_problem_type(profile.problem_type, profile)
        results_df, best_model, best_name, explanation = factory.train_and_compare(
            X_train, X_test, y_train, y_test, profile.problem_type, models
        )
        meta['best_model']      = best_model
        meta['best_model_name'] = best_name

        best_row  = results_df.iloc[0]
        score_col = 'f1' if profile.problem_type != 'regression' else 'r2'
        new_score = best_row.get(score_col, best_row.get('accuracy', 0))

        _add_alert(dataset_id, 'INFO',
                   f'Model retrained. New best: {best_name} (score={new_score:.4f})',
                   'Monitor performance.')
        return jsonify({'best_model': best_name, 'score': round(float(new_score), 4),
                        'explanation': explanation})
    except Exception as exc:
        logger.exception("Retrain failed")
        return jsonify({'error': str(exc)}), 500


@app.route('/api/simulate-batch', methods=['POST'])
def api_simulate_batch():
    dataset_id = _resolve_dataset_id()
    if not dataset_id or dataset_id not in DATASET_REGISTRY:
        return jsonify({'error': 'No active dataset.'}), 404

    meta = DATASET_REGISTRY[dataset_id]
    dm   = meta.get('drift_manager')
    if dm is None:
        return jsonify({'error': 'Pipeline not run yet.'}), 400

    import numpy as np
    baseline  = dm.baseline_df.copy()
    new_batch = baseline + np.random.normal(0, 0.5, baseline.shape)
    result    = dm.check_drift(new_batch)

    if result.alert:
        _add_alert(dataset_id, 'WARNING',
                   f'Drift detected in batch {result.batch_id}. PSI={result.overall_psi:.4f}',
                   'Review drifted features and consider retraining.')

    pr = meta.get('pipeline_results') or {}
    batch_metrics = pr.get('batch_metrics', [])
    batch_metrics.append({
        'batch_id':  result.batch_id,
        'overall_psi': result.overall_psi,
        'n_drifted': len(result.drifted_features),
        'alert':     result.alert,
    })
    pr['batch_metrics'] = batch_metrics
    meta['pipeline_results'] = pr

    return jsonify({
        'batch_id':    result.batch_id,
        'overall_psi': result.overall_psi,
        'alert':       result.alert,
        'drifted':     result.drifted_features,
    })


@app.route('/api/generate-report', methods=['GET'])
def api_generate_report():
    dataset_id = _resolve_dataset_id()
    if not dataset_id or dataset_id not in DATASET_REGISTRY:
        return jsonify({'error': 'No active dataset.'}), 404

    meta    = DATASET_REGISTRY[dataset_id]
    results = meta.get('pipeline_results', {})
    dm      = meta.get('drift_manager')
    alerts  = _ALERTS_STORE.get(dataset_id, [])

    # Build drift dict for report
    drift_dict = {'feature_psi': [], 'kl_scores': {}, 'psi_severity': 'No Drift', 'alert': False}
    if dm and dm._history:
        latest = dm._history[-1]
        drift_dict = {
            'feature_psi': [
                {'feature': r.feature, 'type': r.feature_type, 'psi': r.psi,
                 'severity': r.severity, 'explanation': latest.explanations.get(r.feature, '')}
                for r in latest.psi_report.results
            ],
            'kl_scores':    latest.kl_scores,
            'psi_severity': _psi_severity_label(latest.overall_psi),
            'alert':        latest.alert,
        }

    model_df = pd.DataFrame(results.get('model_results', []))

    try:
        from reports.report_generator import ReportGenerator
        rg       = ReportGenerator()
        filepath = rg.generate_html_report(
            dataset_profile     = meta['profile'],
            model_comparison_df = model_df,
            best_model_name     = meta.get('best_model_name', ''),
            drift_report        = drift_dict,
            performance_summary = results,
            alerts              = alerts,
            dataset_name        = meta['filename'],
        )
        from flask import send_file
        return send_file(filepath, as_attachment=True,
                         download_name=os.path.basename(filepath),
                         mimetype='text/html')
    except Exception as exc:
        logger.exception("Report generation failed")
        return jsonify({'error': str(exc)}), 500


# ---------------------------------------------------------------------------
# Page routes for remaining templates
# ---------------------------------------------------------------------------

@app.route('/model-comparison')
def model_comparison_page():
    active = session.get('active_dataset_id')
    if not active or active not in DATASET_REGISTRY:
        return redirect(url_for('upload_page'))
    return render_template('model_comparison.html')


@app.route('/drift-monitor')
def drift_monitor_page():
    active = session.get('active_dataset_id')
    if not active or active not in DATASET_REGISTRY:
        return redirect(url_for('upload_page'))
    return render_template('drift_monitor.html')


@app.route('/alerts')
def alerts_page():
    return render_template('alerts.html')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ALERTS_STORE: dict = {}   # dataset_id → list of alert dicts
_alert_counter = 0


def _add_alert(dataset_id, level, message, action=''):
    global _alert_counter
    _alert_counter += 1
    _ALERTS_STORE.setdefault(dataset_id, []).append({
        'id':        _alert_counter,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset':   DATASET_REGISTRY.get(dataset_id, {}).get('filename', ''),
        'level':     level,
        'message':   message,
        'action':    action,
        'read':      False,
    })


def _psi_severity_label(psi: float) -> str:
    if psi < 0.1:  return 'No Drift'
    if psi < 0.2:  return 'Moderate'
    return 'Severe'


# Patch api_run_pipeline to store results and generate alerts
_orig_run = app.view_functions['api_run_pipeline']

def _patched_run_pipeline():
    resp = _orig_run()
    try:
        # resp may be a Response object or a (Response, status) tuple
        response_obj = resp[0] if isinstance(resp, tuple) else resp
        if hasattr(response_obj, 'get_json'):
            data = response_obj.get_json(silent=True)
            if data and 'dataset_id' in data:
                did = data['dataset_id']
                if did in DATASET_REGISTRY:
                    DATASET_REGISTRY[did]['pipeline_results'] = data
                    drift = data.get('drift', {})
                    synth = drift.get('synthetic', {})
                    if synth.get('alert'):
                        _add_alert(did, 'CRITICAL',
                                   f"Severe drift in synthetic batch. PSI={synth.get('overall_psi',0):.4f}",
                                   'Retrain model immediately.')
                    stream = drift.get('stream', {})
                    if stream.get('alert'):
                        _add_alert(did, 'WARNING',
                                   f"Drift in incoming stream. PSI={stream.get('overall_psi',0):.4f}",
                                   'Monitor closely and consider retraining.')
                    _add_alert(did, 'INFO',
                               f"Pipeline completed. Best model: {data.get('best_model','N/A')}",
                               'Review model comparison page.')
    except Exception as e:
        logger.warning("Pipeline patch error: %s", e)
    return resp

app.view_functions['api_run_pipeline'] = _patched_run_pipeline


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    # Create required directories
    for d in ['data/uploads', 'data/samples', 'models/saved', 'reports', 'logs']:
        os.makedirs(d, exist_ok=True)

    # File logging
    file_handler = logging.FileHandler('logs/app.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s — %(message)s'))
    logging.getLogger().addHandler(file_handler)

    _ensure_samples()

    # Print route map
    print("\n" + "═" * 60)
    print("  AI Governance Framework — starting on http://localhost:5000")
    print("═" * 60)
    routes = [
        ("GET",    "/",                        "redirect to dashboard or upload"),
        ("GET",    "/upload",                  "dataset upload page"),
        ("GET",    "/dashboard",               "main dashboard"),
        ("GET",    "/model-comparison",        "model comparison page"),
        ("GET",    "/drift-monitor",           "drift monitoring page"),
        ("GET",    "/alerts",                  "alert center"),
        ("POST",   "/api/upload-dataset",      "upload & analyze dataset"),
        ("POST",   "/api/set-target-column",   "override target column"),
        ("POST",   "/api/run-pipeline",        "run full pipeline"),
        ("GET",    "/api/datasets",            "list all datasets"),
        ("DELETE", "/api/datasets/<id>",       "delete dataset"),
        ("GET",    "/api/active-dataset",      "get active dataset"),
        ("POST",   "/api/active-dataset",      "set active dataset"),
        ("GET",    "/api/performance-metrics", "model performance data"),
        ("GET",    "/api/model-comparison",    "model comparison data"),
        ("GET",    "/api/drift-status",        "drift detection results"),
        ("POST",   "/api/simulate-batch",      "simulate next data batch"),
        ("POST",   "/api/retrain",             "retrain best model"),
        ("GET",    "/api/generate-report",     "download HTML report"),
        ("GET",    "/api/alerts",              "list alerts"),
        ("POST",   "/api/alerts/<id>/read",    "mark alert as read"),
    ]
    for method, path, desc in routes:
        print(f"  {method:<7} {path:<35} {desc}")
    print("═" * 60 + "\n")

    app.run(debug=True, port=5000)
