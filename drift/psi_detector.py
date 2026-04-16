import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FeaturePSIResult:
    feature: str
    feature_type: str          # 'numeric' | 'categorical'
    psi: float
    severity: str              # 'No Drift' | 'Slight Drift' | 'Moderate Drift' | 'Severe Drift'
    baseline_stats: dict
    incoming_stats: dict


@dataclass
class PSIReport:
    results: List[FeaturePSIResult]
    overall_psi: float
    drifted_features: List[str]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                'feature':      r.feature,
                'type':         r.feature_type,
                'psi':          round(r.psi, 4),
                'severity':     r.severity,
            }
            for r in self.results
        ])


# ---------------------------------------------------------------------------
# PSIDetector
# ---------------------------------------------------------------------------

class PSIDetector:
    """
    Computes Population Stability Index (PSI) for every feature in a DataFrame.

    Numeric columns  → bucket-based PSI (equal-width bins on baseline)
    Categorical cols → frequency-proportion PSI
    Datetime columns → skipped
    """

    # Standard PSI severity thresholds
    THRESHOLDS = {
        'no_drift':       0.1,
        'slight_drift':   0.2,
        'moderate_drift': 0.25,
    }

    def __init__(self, n_bins: int = 10, epsilon: float = 1e-6):
        self.n_bins  = n_bins
        self.epsilon = epsilon   # avoid log(0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        baseline_df: pd.DataFrame,
        incoming_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> PSIReport:
        """
        Compare incoming_df against baseline_df column by column.
        Returns a PSIReport with per-feature results.
        """
        cols = columns or baseline_df.columns.tolist()
        results: List[FeaturePSIResult] = []

        for col in cols:
            if col not in baseline_df.columns or col not in incoming_df.columns:
                continue

            dtype = baseline_df[col].dtype

            # Skip datetime
            if pd.api.types.is_datetime64_any_dtype(dtype):
                continue

            if pd.api.types.is_numeric_dtype(dtype):
                result = self._numeric_psi(col, baseline_df[col], incoming_df[col])
            else:
                result = self._categorical_psi(col, baseline_df[col], incoming_df[col])

            results.append(result)

        overall_psi    = float(np.mean([r.psi for r in results])) if results else 0.0
        drifted        = [r.feature for r in results if r.severity != 'No Drift']

        return PSIReport(results=results, overall_psi=overall_psi, drifted_features=drifted)

    # ------------------------------------------------------------------
    # Numeric PSI  (bucket-based)
    # ------------------------------------------------------------------

    def _numeric_psi(
        self, col: str, baseline: pd.Series, incoming: pd.Series
    ) -> FeaturePSIResult:
        base_clean = baseline.dropna()
        inc_clean  = incoming.dropna()

        # Build bins from baseline distribution
        _, bin_edges = np.histogram(base_clean, bins=self.n_bins)
        bin_edges[0]  -= self.epsilon   # include minimum value
        bin_edges[-1] += self.epsilon

        base_counts, _ = np.histogram(base_clean, bins=bin_edges)
        inc_counts,  _ = np.histogram(inc_clean,  bins=bin_edges)

        base_pct = base_counts / (len(base_clean) + self.epsilon)
        inc_pct  = inc_counts  / (len(inc_clean)  + self.epsilon)

        # Avoid zero proportions
        base_pct = np.where(base_pct == 0, self.epsilon, base_pct)
        inc_pct  = np.where(inc_pct  == 0, self.epsilon, inc_pct)

        psi = float(np.sum((inc_pct - base_pct) * np.log(inc_pct / base_pct)))

        baseline_stats = {
            'mean':   round(float(base_clean.mean()), 4),
            'std':    round(float(base_clean.std()),  4),
            'median': round(float(base_clean.median()), 4),
        }
        incoming_stats = {
            'mean':   round(float(inc_clean.mean()), 4),
            'std':    round(float(inc_clean.std()),  4),
            'median': round(float(inc_clean.median()), 4),
        }

        return FeaturePSIResult(
            feature=col,
            feature_type='numeric',
            psi=psi,
            severity=self._severity(psi),
            baseline_stats=baseline_stats,
            incoming_stats=incoming_stats,
        )

    # ------------------------------------------------------------------
    # Categorical PSI  (frequency-proportion based)
    # ------------------------------------------------------------------

    def _categorical_psi(
        self, col: str, baseline: pd.Series, incoming: pd.Series
    ) -> FeaturePSIResult:
        base_clean = baseline.dropna().astype(str)
        inc_clean  = incoming.dropna().astype(str)

        all_cats = set(base_clean.unique()) | set(inc_clean.unique())

        base_freq = base_clean.value_counts(normalize=True).to_dict()
        inc_freq  = inc_clean.value_counts(normalize=True).to_dict()

        psi = 0.0
        for cat in all_cats:
            b = base_freq.get(cat, self.epsilon)
            i = inc_freq.get(cat,  self.epsilon)
            psi += (i - b) * np.log(i / b)

        baseline_stats = {
            'top_categories': base_clean.value_counts().head(5).to_dict(),
            'n_unique':       int(base_clean.nunique()),
        }
        incoming_stats = {
            'top_categories': inc_clean.value_counts().head(5).to_dict(),
            'n_unique':       int(inc_clean.nunique()),
        }

        return FeaturePSIResult(
            feature=col,
            feature_type='categorical',
            psi=float(psi),
            severity=self._severity(float(psi)),
            baseline_stats=baseline_stats,
            incoming_stats=incoming_stats,
        )

    # ------------------------------------------------------------------
    # Severity label
    # ------------------------------------------------------------------

    def _severity(self, psi: float) -> str:
        if psi < self.THRESHOLDS['no_drift']:
            return 'No Drift'
        if psi < self.THRESHOLDS['slight_drift']:
            return 'Slight Drift'
        if psi < self.THRESHOLDS['moderate_drift']:
            return 'Moderate Drift'
        return 'Severe Drift'
