"""
Churn Prediction and Customer Retention Risk Classifier.
Predicts account attrition probabilities, pinpoints behavioral risk factors,
and generates prioritized customer retention watchlists with dollar impact.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from data_intelligence.types import ModelType, ModelResult, DatasetProfile


class ChurnPredictionEngine:
    """
    Automated churn classification and risk scoring engine.
    """

    CHURN_KEYWORDS = ["churn", "churned", "cancelled", "canceled", "attrition", "is_churn"]

    def __init__(self, profile: DatasetProfile):
        self.profile = profile

    def run(
        self,
        df: pd.DataFrame,
        churn_col: Optional[str] = None,
        customer_col: Optional[str] = None,
        revenue_col: Optional[str] = None,
    ) -> ModelResult:
        """
        Execute churn classification and feature importance pipeline.
        """
        start_time = time.time()
        c_col = customer_col or self.profile.detected_customer_id_col
        rev_col = revenue_col or self.profile.detected_target_col

        # Detect churn column
        target_col = churn_col or self._detect_churn_column(df)
        work_df = df.copy()

        # If no explicit churn column is present, formulate implicit churn from recency if possible
        if not target_col:
            dt_col = self.profile.detected_datetime_col
            if dt_col and dt_col in work_df.columns and c_col and c_col in work_df.columns:
                target_col, work_df = self._create_implicit_churn(work_df, c_col, dt_col)
            else:
                return ModelResult(
                    model_type=ModelType.CHURN_PREDICTION,
                    model_name="Supervised Churn Risk Classifier",
                    status="skipped",
                    insights=["Churn modeling skipped: No explicit churn target or transaction timestamps found."],
                    error_message="Churn target not detected.",
                    execution_time_seconds=round(time.time() - start_time, 2),
                )

        try:
            # Prepare feature matrix
            y = work_df[target_col].astype(int)
            churn_rate = float(y.mean())

            if y.nunique() < 2 or churn_rate < 0.01 or churn_rate > 0.99:
                return ModelResult(
                    model_type=ModelType.CHURN_PREDICTION,
                    model_name="Supervised Churn Risk Classifier",
                    status="warning",
                    insights=[f"Churn target exhibits insufficient variance (churn rate: {churn_rate * 100:.1f}%)."],
                    execution_time_seconds=round(time.time() - start_time, 2),
                )

            # Drop non-feature columns
            drop_cols = [target_col]
            if c_col and c_col in work_df.columns:
                drop_cols.append(c_col)
            for c in work_df.columns:
                if any(kw in c.lower() for kw in ["date", "time", "id", "name", "email", "phone"]):
                    if c not in drop_cols:
                        drop_cols.append(c)

            features_df = work_df.drop(columns=[c for c in drop_cols if c in work_df.columns])

            # Numeric encoding
            features_df = pd.get_dummies(features_df, drop_first=True)
            # Impute any remaining nulls
            features_df = features_df.fillna(features_df.median())

            if features_df.shape[1] < 2:
                return ModelResult(
                    model_type=ModelType.CHURN_PREDICTION,
                    model_name="Supervised Churn Risk Classifier",
                    status="warning",
                    insights=["Insufficient behavioral features to train churn predictor."],
                    execution_time_seconds=round(time.time() - start_time, 2),
                )

            X = features_df.values
            feature_names = list(features_df.columns)

            # Stratified train-test split

        return None
