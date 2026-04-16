import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from scipy.stats import entropy as kl_divergence

from drift.psi_detector import PSIDetector, PSIReport, FeaturePSIResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds (user-configurable)
# ---------------------------------------------------------------------------

@dataclass
class DriftThresholds:
    psi_slight:   float = 0.1
    psi_moderate: float = 0.2
    psi_severe:   float = 0.25
    kl_threshold: float = 0.1   # KL divergence alert threshold


# ---------------------------------------------------------------------------
# Per-batch drift record
# ---------------------------------------------------------------------------

@dataclass
class DriftBatchResult:
    batch_id:          int
    psi_report:        PSIReport
    kl_scores:         Dict[str, float]
    drifted_features:  List[str]
    explanations:      Dict[str, str]   # feature_name → human-readable text
    overall_psi:       float
    alert:             bool


# ---------------------------------------------------------------------------
# DriftManager
# ---------------------------------------------------------------------------

class DriftManager:
    """
    Stores a baseline DataFrame and compares every incoming batch against it.

    Thresholds for PSI and KL divergence are fully configurable so the UI
    can expose them as sliders / inputs.
    """

    def __init__(
        self,
        baseline_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        thresholds: Optional[DriftThresholds] = None,
        n_bins: int = 10,
    ):
        self.baseline_df      = baseline_df.copy()
        self.feature_columns  = feature_columns or baseline_df.columns.tolist()
        self.thresholds       = thresholds or DriftThresholds()
        self._batch_counter   = 0
        self._history: List[DriftBatchResult] = []

        # Build PSI detector with current thresholds
        self._psi_detector = self._build_psi_detector(n_bins)

    # ------------------------------------------------------------------
    # Threshold update (called from UI)
    # ------------------------------------------------------------------

    def update_thresholds(
        self,
        psi_slight:   Optional[float] = None,
        psi_moderate: Optional[float] = None,
        psi_severe:   Optional[float] = None,
        kl_threshold: Optional[float] = None,
    ) -> None:
        """Allow the UI to update any threshold at runtime."""
        if psi_slight   is not None: self.thresholds.psi_slight   = psi_slight
        if psi_moderate is not None: self.thresholds.psi_moderate = psi_moderate
        if psi_severe   is not None: self.thresholds.psi_severe   = psi_severe
        if kl_threshold is not None: self.thresholds.kl_threshold = kl_threshold

        # Sync thresholds into the PSI detector
        self._psi_detector.THRESHOLDS = {
            'no_drift':       self.thresholds.psi_slight,
            'slight_drift':   self.thresholds.psi_moderate,
            'moderate_drift': self.thresholds.psi_severe,
        }
        logger.info("Drift thresholds updated: %s", self.thresholds)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def check_drift(self, incoming_df: pd.DataFrame) -> DriftBatchResult:
        """
        Compare incoming_df against the stored baseline.
        Returns a DriftBatchResult with PSI scores, KL scores, and explanations.
        """
        self._batch_counter += 1

        psi_report = self._psi_detector.detect_drift(
            self.baseline_df, incoming_df, columns=self.feature_columns
        )
        kl_scores  = self._compute_kl_scores(incoming_df)

        drifted    = psi_report.drifted_features
        explanations = {
            r.feature: self.explain_drift(r.feature, r.psi, r, incoming_df)
            for r in psi_report.results
            if r.severity != 'No Drift'
        }

        alert = (
            psi_report.overall_psi >= self.thresholds.psi_severe
            or any(v >= self.thresholds.kl_threshold for v in kl_scores.values())
        )

        result = DriftBatchResult(
            batch_id=self._batch_counter,
            psi_report=psi_report,
            kl_scores=kl_scores,
            drifted_features=drifted,
            explanations=explanations,
            overall_psi=round(psi_report.overall_psi, 4),
            alert=alert,
        )
        self._history.append(result)
        logger.info(
            "Batch %d — overall PSI=%.4f, drifted features: %s",
            self._batch_counter, psi_report.overall_psi, drifted,
        )
        return result

    # ------------------------------------------------------------------
    # Human-readable explanation
    # ------------------------------------------------------------------

    def explain_drift(
        self,
        feature_name: str,
        psi_score: float,
        result: Optional[FeaturePSIResult] = None,
        incoming_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Generate a human-readable drift explanation for a single feature.

        Works standalone (just feature_name + psi_score) or with richer
        context when result / incoming_df are provided.
        """
        severity = self._severity_label(psi_score)
        base_line = f"Feature '{feature_name}' has PSI={psi_score:.4f} ({severity})."

        if result is None:
            return base_line

        if result.feature_type == 'numeric':
            base_mean = result.baseline_stats.get('mean', 'N/A')
            inc_mean  = result.incoming_stats.get('mean', 'N/A')
            base_std  = result.baseline_stats.get('std', 'N/A')
            inc_std   = result.incoming_stats.get('std', 'N/A')

            direction = ''
            if isinstance(base_mean, (int, float)) and isinstance(inc_mean, (int, float)):
                diff = inc_mean - base_mean
                direction = (
                    f" The average value shifted from {base_mean} in baseline "
                    f"to {inc_mean} in incoming data "
                    f"({'increased' if diff > 0 else 'decreased'} by {abs(round(diff, 4))})."
                )

            spread = ''
            if isinstance(base_std, (int, float)) and isinstance(inc_std, (int, float)):
                if abs(inc_std - base_std) / (base_std + 1e-9) > 0.2:
                    spread = (
                        f" Variability also changed (std: {base_std} → {inc_std}),"
                        " suggesting a shift in data spread."
                    )

            suggestion = self._numeric_suggestion(severity)
            return f"{base_line}{direction}{spread} {suggestion}"

        else:  # categorical
            base_top = result.baseline_stats.get('top_categories', {})
            inc_top  = result.incoming_stats.get('top_categories', {})
            base_n   = result.baseline_stats.get('n_unique', '?')
            inc_n    = result.incoming_stats.get('n_unique', '?')

            new_cats = set(inc_top.keys()) - set(base_top.keys())
            gone_cats = set(base_top.keys()) - set(inc_top.keys())

            detail = f" Unique categories: {base_n} (baseline) → {inc_n} (incoming)."
            if new_cats:
                detail += f" New categories appeared: {', '.join(list(new_cats)[:5])}."
            if gone_cats:
                detail += f" Categories no longer present: {', '.join(list(gone_cats)[:5])}."

            suggestion = self._categorical_suggestion(severity)
            return f"{base_line}{detail} {suggestion}"

    # ------------------------------------------------------------------
    # History / reporting
    # ------------------------------------------------------------------

    def get_history(self) -> List[DriftBatchResult]:
        return self._history

    def get_drift_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising drift across all batches."""
        rows = []
        for b in self._history:
            rows.append({
                'batch_id':     b.batch_id,
                'overall_psi':  b.overall_psi,
                'n_drifted':    len(b.drifted_features),
                'drifted':      ', '.join(b.drifted_features),
                'alert':        b.alert,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # KL divergence (numeric features only)
    # ------------------------------------------------------------------

    def _compute_kl_scores(self, incoming_df: pd.DataFrame) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for col in self.feature_columns:
            if col not in self.baseline_df.columns or col not in incoming_df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.baseline_df[col]):
                continue

            base_vals = self.baseline_df[col].dropna().values
            inc_vals  = incoming_df[col].dropna().values
            if len(base_vals) == 0 or len(inc_vals) == 0:
                continue

            # Build shared histogram
            combined = np.concatenate([base_vals, inc_vals])
            bins = np.histogram_bin_edges(combined, bins=10)
            eps  = 1e-9

            base_hist, _ = np.histogram(base_vals, bins=bins, density=True)
            inc_hist,  _ = np.histogram(inc_vals,  bins=bins, density=True)

            base_hist = base_hist + eps
            inc_hist  = inc_hist  + eps
            base_hist /= base_hist.sum()
            inc_hist  /= inc_hist.sum()

            scores[col] = round(float(kl_divergence(base_hist, inc_hist)), 4)

        return scores

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_psi_detector(self, n_bins: int) -> PSIDetector:
        detector = PSIDetector(n_bins=n_bins)
        detector.THRESHOLDS = {
            'no_drift':       self.thresholds.psi_slight,
            'slight_drift':   self.thresholds.psi_moderate,
            'moderate_drift': self.thresholds.psi_severe,
        }
        return detector

    def _severity_label(self, psi: float) -> str:
        t = self.thresholds
        if psi < t.psi_slight:   return 'No Drift'
        if psi < t.psi_moderate: return 'Slight Drift'
        if psi < t.psi_severe:   return 'Moderate Drift'
        return 'Severe Drift'

    @staticmethod
    def _numeric_suggestion(severity: str) -> str:
        if severity == 'Severe Drift':
            return (
                "This may indicate a significant change in the underlying population. "
                "Consider retraining the model with recent data."
            )
        if severity == 'Moderate Drift':
            return (
                "Monitor this feature closely. "
                "Partial retraining or recalibration may be needed."
            )
        return "Minor shift detected. Continue monitoring."

    @staticmethod
    def _categorical_suggestion(severity: str) -> str:
        if severity == 'Severe Drift':
            return (
                "The category distribution has changed substantially. "
                "Review data pipeline and consider model retraining."
            )
        if severity == 'Moderate Drift':
            return (
                "Noticeable category shift. "
                "Validate upstream data sources and monitor model performance."
            )
        return "Slight category shift. Continue monitoring."
