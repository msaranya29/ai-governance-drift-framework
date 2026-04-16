import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ModelFactory
# ---------------------------------------------------------------------------

class ModelFactory:

    # ------------------------------------------------------------------
    # Public: build model catalogue
    # ------------------------------------------------------------------

    def get_models_for_problem_type(
        self,
        problem_type: str,
        dataset_profile,          # DatasetProfile dataclass
    ) -> Dict[str, Any]:
        """Return a dict of {model_name: estimator} for the given problem type."""
        n_rows = dataset_profile.n_rows
        n_cols = dataset_profile.n_columns

        if problem_type in ('binary_classification', 'multiclass_classification'):
            return self._classification_models(n_rows, n_cols)
        elif problem_type == 'regression':
            return self._regression_models(n_rows)
        else:
            raise ValueError(f"Unknown problem_type: {problem_type}")

    # ------------------------------------------------------------------
    # Classification catalogue
    # ------------------------------------------------------------------

    def _classification_models(self, n_rows: int, n_cols: int) -> Dict[str, Any]:
        lr  = LogisticRegression(max_iter=1000, random_state=42)
        rf  = RandomForestClassifier(n_estimators=100, random_state=42)
        dt  = DecisionTreeClassifier(random_state=42)
        gb  = GradientBoostingClassifier(n_estimators=100, random_state=42)
        nb  = GaussianNB()
        ada = AdaBoostClassifier(n_estimators=100, random_state=42)

        models: Dict[str, Any] = {
            'Logistic Regression':     lr,
            'Random Forest':           rf,
            'Decision Tree':           dt,
            'Gradient Boosting':       gb,
            'Naive Bayes':             nb,
            'AdaBoost':                ada,
        }

        if n_rows < 10_000:
            models['SVM'] = SVC(probability=True, random_state=42)
        if n_cols < 20:
            models['KNN'] = KNeighborsClassifier()

        # Ensemble / combination models
        voting_hard = VotingClassifier(
            estimators=[('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                        ('lr', LogisticRegression(max_iter=1000, random_state=42))],
            voting='hard',
        )
        voting_soft = VotingClassifier(
            estimators=[('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                        ('lr', LogisticRegression(max_iter=1000, random_state=42))],
            voting='soft',
        )
        stacking = StackingClassifier(
            estimators=[('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                        ('lr', LogisticRegression(max_iter=1000, random_state=42))],
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5,
        )
        bagging = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=50,
            random_state=42,
        )

        models['Voting (Hard)']   = voting_hard
        models['Voting (Soft)']   = voting_soft
        models['Stacking']        = stacking
        models['Bagging']         = bagging

        return models

    # ------------------------------------------------------------------
    # Regression catalogue
    # ------------------------------------------------------------------

    def _regression_models(self, n_rows: int) -> Dict[str, Any]:
        lr  = LinearRegression()
        rid = Ridge(random_state=42)
        las = Lasso(random_state=42)
        rf  = RandomForestRegressor(n_estimators=100, random_state=42)
        gb  = GradientBoostingRegressor(n_estimators=100, random_state=42)
        dt  = DecisionTreeRegressor(random_state=42)

        models: Dict[str, Any] = {
            'Linear Regression':        lr,
            'Ridge Regression':         rid,
            'Lasso Regression':         las,
            'Random Forest':            rf,
            'Gradient Boosting':        gb,
            'Decision Tree':            dt,
        }

        if n_rows < 10_000:
            models['SVR'] = SVR()

        voting_reg = VotingRegressor(
            estimators=[('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                        ('lr', LinearRegression())],
        )
        stacking_reg = StackingRegressor(
            estimators=[('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))],
            final_estimator=Ridge(random_state=42),
            cv=5,
        )
        bagging_reg = BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=42),
            n_estimators=50,
            random_state=42,
        )

        models['Voting Regressor']   = voting_reg
        models['Stacking Regressor'] = stacking_reg
        models['Bagging Regressor']  = bagging_reg

        return models

    # ------------------------------------------------------------------
    # Train & compare
    # ------------------------------------------------------------------

    def train_and_compare(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        problem_type: str,
        models: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, Any, str, str]:
        """
        Train all models, compute metrics, return:
          (results_df, best_estimator, best_model_name, explanation_text)
        """
        is_classification = problem_type in ('binary_classification', 'multiclass_classification')
        is_multiclass     = problem_type == 'multiclass_classification'
        results = []

        for name, model in (models or {}).items():
            row = self._train_single(
                name, model, X_train, X_test, y_train, y_test,
                is_classification, is_multiclass,
            )
            results.append(row)
            logger.info("Trained %s — %s", name, row)

        df = pd.DataFrame(results)
        sort_col = 'f1' if is_classification else 'r2'
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

        best_row  = df.iloc[0]
        best_name = best_row['model']
        best_score = best_row[sort_col]

        # Find best individual (non-ensemble) model for the log message
        ensemble_keywords = {'Voting', 'Stacking', 'Bagging', 'AdaBoost'}
        individual_df = df[~df['model'].apply(
            lambda m: any(k in m for k in ensemble_keywords)
        )]

        explanation = self._build_explanation(
            best_name, best_score, individual_df, sort_col, is_classification
        )
        logger.info(explanation)

        # Re-fit best model on full training data so caller gets a ready estimator
        best_model = (models or {})[best_name]
        best_model.fit(X_train, y_train)

        return df, best_model, best_name, explanation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_single(
        self,
        name: str,
        model: Any,
        X_train, X_test, y_train, y_test,
        is_classification: bool,
        is_multiclass: bool,
    ) -> dict:
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            elapsed = round(time.time() - t0, 4)

            if is_classification:
                avg = 'weighted' if is_multiclass else 'binary'
                row = {
                    'model':     name,
                    'accuracy':  round(accuracy_score(y_test, y_pred), 4),
                    'precision': round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
                    'recall':    round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
                    'f1':        round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
                    'roc_auc':   self._safe_roc_auc(model, X_test, y_test, is_multiclass),
                    'train_time_s': elapsed,
                }
            else:
                row = {
                    'model':     name,
                    'mae':       round(mean_absolute_error(y_test, y_pred), 4),
                    'mse':       round(mean_squared_error(y_test, y_pred), 4),
                    'rmse':      round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                    'r2':        round(r2_score(y_test, y_pred), 4),
                    'train_time_s': elapsed,
                }
        except Exception as exc:
            logger.warning("Model %s failed: %s", name, exc)
            row = {'model': name, 'error': str(exc), 'train_time_s': -1}
            if is_classification:
                row.update({'accuracy': np.nan, 'precision': np.nan,
                            'recall': np.nan, 'f1': np.nan, 'roc_auc': np.nan})
            else:
                row.update({'mae': np.nan, 'mse': np.nan, 'rmse': np.nan, 'r2': np.nan})

        return row

    def _safe_roc_auc(self, model, X_test, y_test, is_multiclass: bool) -> float:
        try:
            multi_kw = 'ovr' if is_multiclass else 'raise'
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)
                if is_multiclass:
                    return round(roc_auc_score(y_test, proba, multi_class='ovr', average='weighted'), 4)
                return round(roc_auc_score(y_test, proba[:, 1]), 4)
        except Exception:
            pass
        return np.nan

    def _build_explanation(
        self,
        best_name: str,
        best_score: float,
        individual_df: pd.DataFrame,
        sort_col: str,
        is_classification: bool,
    ) -> str:
        metric_label = 'F1' if is_classification else 'R²'
        ensemble_keywords = {'Voting', 'Stacking', 'Bagging', 'AdaBoost'}
        is_ensemble = any(k in best_name for k in ensemble_keywords)

        if is_ensemble and not individual_df.empty:
            best_ind_row   = individual_df.iloc[0]
            best_ind_name  = best_ind_row['model']
            best_ind_score = best_ind_row[sort_col]
            return (
                f"{best_name} outperformed all individual models with "
                f"{metric_label}={best_score:.4f} vs best individual "
                f"{best_ind_name} {metric_label}={best_ind_score:.4f}."
            )

        if not individual_df.empty and individual_df.iloc[0]['model'] != best_name:
            runner_up      = individual_df.iloc[1] if len(individual_df) > 1 else None
            runner_text    = (
                f" (runner-up: {runner_up['model']} {metric_label}={runner_up[sort_col]:.4f})"
                if runner_up is not None else ""
            )
            return (
                f"{best_name} achieved the best {metric_label}={best_score:.4f}"
                f"{runner_text}."
            )

        return f"{best_name} is the best model with {metric_label}={best_score:.4f}."
