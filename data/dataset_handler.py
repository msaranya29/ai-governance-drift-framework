import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


@dataclass
class DatasetProfile:
    n_rows: int
    n_columns: int
    feature_types: Dict[str, str]
    missing_values: Dict[str, int]
    class_distribution: Dict
    problem_type: str
    recommended_models: List[str]


class DatasetHandler:
    KNOWN_TARGET_NAMES = {'target', 'label', 'class', 'output', 'y', 'churn', 'fraud', 'diagnosis'}

    def __init__(self, filepath: str, target_column: Optional[str] = None):
        self.filepath = filepath
        self.target_column = target_column
        self._profile: Optional[DatasetProfile] = None

    # ------------------------------------------------------------------
    # Target detection
    # ------------------------------------------------------------------
    def auto_detect_target(self, df: pd.DataFrame) -> str:
        for col in df.columns:
            if col.lower() in self.KNOWN_TARGET_NAMES:
                return col
        return df.columns[-1]

    # ------------------------------------------------------------------
    # Dataset analysis
    # ------------------------------------------------------------------
    def analyze_dataset(self, df: pd.DataFrame) -> DatasetProfile:
        target = self.target_column or self.auto_detect_target(df)

        feature_types: Dict[str, str] = {}
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_types[col] = 'datetime'
            elif pd.api.types.is_numeric_dtype(df[col]):
                feature_types[col] = 'numeric'
            else:
                feature_types[col] = 'categorical'

        missing_values = {col: int(df[col].isna().sum()) for col in df.columns}
        class_distribution = df[target].value_counts().to_dict()

        problem_type = self._detect_problem_type(df[target])
        recommended_models = self._recommend_models(problem_type)

        self._profile = DatasetProfile(
            n_rows=len(df),
            n_columns=len(df.columns),
            feature_types=feature_types,
            missing_values=missing_values,
            class_distribution=class_distribution,
            problem_type=problem_type,
            recommended_models=recommended_models,
        )
        return self._profile

    def _detect_problem_type(self, target_series: pd.Series) -> str:
        n_unique = target_series.nunique()
        if pd.api.types.is_numeric_dtype(target_series) and n_unique > 20:
            return 'regression'
        if n_unique <= 2:
            return 'binary_classification'
        return 'multiclass_classification'

    def _recommend_models(self, problem_type: str) -> List[str]:
        base = ['Random Forest', 'Gradient Boosting', 'XGBoost']
        if problem_type == 'regression':
            return base + ['Linear Regression', 'Ridge', 'SVR']
        return base + ['Logistic Regression', 'SVM', 'KNN']

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocess(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler]:
        df = df.copy()

        # Drop columns with >50% missing
        threshold = len(df) * 0.5
        df = df.dropna(thresh=threshold, axis=1)

        # Separate features and target
        y_series = df[target_column]
        X_df = df.drop(columns=[target_column])

        # Encode target if categorical
        if not pd.api.types.is_numeric_dtype(y_series):
            le = LabelEncoder()
            y = le.fit_transform(y_series)
        else:
            y = y_series.to_numpy()

        # Fill missing values
        numeric_cols = X_df.select_dtypes(include='number').columns.tolist()
        cat_cols = X_df.select_dtypes(exclude='number').columns.tolist()

        for col in numeric_cols:
            X_df[col] = X_df[col].fillna(X_df[col].median())
        for col in cat_cols:
            mode_val = X_df[col].mode()
            X_df[col] = X_df[col].fillna(mode_val[0] if not mode_val.empty else 'unknown')

        # Label-encode categorical columns
        for col in cat_cols:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))

        feature_names = X_df.columns.tolist()

        # Scale numeric features
        scaler = StandardScaler()
        X_df[numeric_cols] = scaler.fit_transform(X_df[numeric_cols])

        return X_df.to_numpy(), y, feature_names, scaler

    # ------------------------------------------------------------------
    # Train/stream/drift split
    # ------------------------------------------------------------------
    def split_with_drift_simulation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        baseline_ratio: float = 0.6,
    ) -> Tuple:
        n = len(X)
        baseline_end = int(n * baseline_ratio)
        drift_start = int(n * 0.8)

        X_baseline, y_baseline = X[:baseline_end], y[:baseline_end]
        X_stream, y_stream = X[baseline_end:], y[baseline_end:]

        # Train/test split from baseline
        X_train, X_test, y_train, y_test = train_test_split(
            X_baseline, y_baseline, test_size=0.2, random_state=42
        )

        # Synthetic drift: last 20% + Gaussian noise on numeric features
        X_drifted = X[drift_start:].copy().astype(float)
        X_drifted += np.random.normal(loc=0, scale=1.5, size=X_drifted.shape)
        y_drifted = y[drift_start:]

        return X_train, X_test, y_train, y_test, X_stream, y_stream, X_drifted, y_drifted

    # ------------------------------------------------------------------
    # Summary card
    # ------------------------------------------------------------------
    def get_dataset_summary_card(self) -> dict:
        if self._profile is None:
            raise RuntimeError("Call analyze_dataset() before get_dataset_summary_card().")

        profile = self._profile
        target = self.target_column or ''

        total = sum(profile.class_distribution.values()) or 1
        class_balance = {
            str(k): round(v / total * 100, 2)
            for k, v in profile.class_distribution.items()
        }

        total_cells = profile.n_rows * profile.n_columns or 1
        missing_pct = round(
            sum(profile.missing_values.values()) / total_cells * 100, 2
        )

        import os
        dataset_name = os.path.basename(self.filepath)

        return {
            'dataset_name': dataset_name,
            'n_rows': profile.n_rows,
            'n_cols': profile.n_columns,
            'target_column': target,
            'problem_type': profile.problem_type,
            'class_balance': class_balance,
            'missing_data_%': missing_pct,
        }
