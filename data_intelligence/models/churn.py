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
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )

            # Train Random Forest Classifier
            rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            rf.fit(X_train, y_train)

            # Predictions and probability scores
            test_probs = rf.predict_proba(X_test)[:, 1]
            test_preds = (test_probs >= 0.5).astype(int)

            roc_auc = float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.5
            precision = float(precision_score(y_test, test_preds, zero_division=0))
            recall = float(recall_score(y_test, test_preds, zero_division=0))
            f1 = float(f1_score(y_test, test_preds, zero_division=0))

            # Feature Importance
            importances = rf.feature_importances_
            feature_imp_sorted = sorted(
                zip(feature_names, importances), key=lambda x: x[1], reverse=True
            )
            top_factors = [{"feature": f, "importance_pct": round(float(imp) * 100, 1)} for f, imp in feature_imp_sorted[:6]]

            # Churn probability on full dataset for watchlist
            full_probs = rf.predict_proba(X)[:, 1]
            work_df["churn_probability"] = np.round(full_probs, 3)

            # Identify high risk accounts (prob >= 0.65 and not already churned)
            high_risk_filter = (work_df["churn_probability"] >= 0.65) & (work_df[target_col] == 0)
            high_risk_df = work_df[high_risk_filter]

            # Quantify dollars at risk
            total_at_risk_dollars = 0.0
            if rev_col and rev_col in work_df.columns:
                total_at_risk_dollars = float(high_risk_df[rev_col].sum())
            elif "mrr" in work_df.columns:
                total_at_risk_dollars = float(high_risk_df["mrr"].sum() * 12.0)

            # High-risk watchlist table
            id_col_to_use = c_col if c_col and c_col in work_df.columns else "index"
            watchlist_cols = [id_col_to_use, "churn_probability"]
            if rev_col and rev_col in work_df.columns:
                watchlist_cols.append(rev_col)
            elif "mrr" in work_df.columns:
                watchlist_cols.append("mrr")

            watchlist = high_risk_df.sort_values(by="churn_probability", ascending=False)[
                [c for c in watchlist_cols if c in high_risk_df.columns]
            ].head(25).to_dict(orient="records")

            insights = [
                f"Overall baseline churn rate is {churn_rate * 100:.1f}%. Model predictive performance achieved an ROC-AUC of {roc_auc:.2f} (Recall: {recall * 100:.1f}%).",
                f"Top churn leading indicator: '{top_factors[0]['feature']}' (explaining {top_factors[0]['importance_pct']}% of predictive variance), followed by '{top_factors[1]['feature'] if len(top_factors) > 1 else 'N/A'}'.",
            ]

            if total_at_risk_dollars > 0:
                insights.append(
                    f"Financial Risk Exposure: {len(high_risk_df)} active accounts currently exhibit high churn signals (≥65% probability), representing ${total_at_risk_dollars:,.2f} in revenue at immediate risk."
                )

            return ModelResult(
                model_type=ModelType.CHURN_PREDICTION,
                model_name="Supervised Churn Risk Classifier (Random Forest)",
                status="success",
                metrics={
                    "roc_auc": round(roc_auc, 3),
                    "precision": round(precision, 3),
                    "recall": round(recall, 3),
                    "f1_score": round(f1, 3),
                    "baseline_churn_rate_pct": round(churn_rate * 100, 1),
                },
                insights=insights,
                quantitative_findings={
                    "high_risk_accounts_count": len(high_risk_df),
                    "revenue_at_risk_usd": round(total_at_risk_dollars, 2),
                    "primary_churn_driver": top_factors[0]["feature"],
                },
                tables={
                    "top_churn_drivers": top_factors,
                    "high_risk_watchlist": watchlist,
                },
                charts_data={
                    "driver_names": [f["feature"] for f in top_factors],
                    "driver_importances": [f["importance_pct"] for f in top_factors],
                    "risk_distribution": [
                        {"range": "0-20% (Safe)", "count": int((full_probs < 0.2).sum())},
                        {"range": "20-50% (Moderate)", "count": int(((full_probs >= 0.2) & (full_probs < 0.5)).sum())},
                        {"range": "50-70% (Elevated)", "count": int(((full_probs >= 0.5) & (full_probs < 0.7)).sum())},
                        {"range": "70-100% (Critical)", "count": int((full_probs >= 0.7).sum())},
                    ],
                },
                execution_time_seconds=round(time.time() - start_time, 2),
            )
        except Exception as e:
            return ModelResult(
                model_type=ModelType.CHURN_PREDICTION,
                model_name="Supervised Churn Risk Classifier",
                status="error",
                error_message=str(e),

        return None
