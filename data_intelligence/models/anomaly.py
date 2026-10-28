"""
Operational and Financial Anomaly Detection Engine for SMBs.
Utilizes Isolation Forests and statistical IQR bounds to surface hidden
revenue leaks, operational cost spikes, inventory shrinkage, and margin compression.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from data_intelligence.types import ModelType, ModelResult, DatasetProfile


class AnomalyDetectionEngine:
    """
    Automated anomaly and operational outlier detection.
    """

    def __init__(self, profile: DatasetProfile, contamination: float = 0.03):
        self.profile = profile
        self.contamination = contamination

    def run(self, df: pd.DataFrame) -> ModelResult:
        """
        Execute multidimensional anomaly detection pipeline.
        """
        start_time = time.time()
        work_df = df.copy()

        # Select continuous numeric features
        num_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
        # Drop ID-like columns
        feature_cols = [
            c for c in num_cols
            if not any(id_kw in c.lower() for id_kw in ["id", "index", "code", "zip", "phone"])
            and work_df[c].nunique() > 5
        ]

        if len(feature_cols) < 2 or len(work_df) < 25:
            return ModelResult(
                model_type=ModelType.ANOMALY_DETECTION,
                model_name="Isolation Forest Anomaly Radar",
                status="skipped",
                insights=["Anomaly detection skipped: Insufficient numerical feature variance or records (< 25 rows)."],
                execution_time_seconds=round(time.time() - start_time, 2),
            )

        try:
            X = work_df[feature_cols].fillna(work_df[feature_cols].median())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
            )
            preds = iso_forest.fit_predict(X_scaled)  # -1 for anomaly, 1 for normal
            anomaly_scores = -iso_forest.decision_function(X_scaled)  # Higher is more anomalous

            work_df["is_anomaly"] = (preds == -1).astype(int)
            work_df["anomaly_score"] = np.round(anomaly_scores, 3)

            n_anomalies = int(work_df["is_anomaly"].sum())
            anomaly_rate_pct = round((n_anomalies / len(work_df)) * 100, 2)

            # Extract top anomalous records
            anomalies_df = work_df[work_df["is_anomaly"] == 1].sort_values(
                by="anomaly_score", ascending=False
            )

            # Build detailed alert entries
            date_col = self.profile.detected_datetime_col
            id_col = self.profile.detected_customer_id_col or "order_id"

            alert_items = []
            for _, row in anomalies_df.head(20).iterrows():
                # Identify which features deviated most
                deviations = []
                for col in feature_cols[:4]:
                    col_mean = work_df[col].mean()
                    col_std = work_df[col].std() or 1.0
                    val = row[col]
                    z_val = (val - col_mean) / col_std
                    if abs(z_val) > 2.0:
                        deviations.append(f"{col}: {val} ({'+' if z_val > 0 else ''}{z_val:.1f}σ vs mean)")

                identifier_str = str(row[id_col]) if id_col in row else f"Row {row.name}"
                date_str = str(row[date_col]) if date_col and date_col in row else "N/A"

                severity = "Critical" if row["anomaly_score"] > 0.15 else "High"
                explanation = "; ".join(deviations) if deviations else "Unusual multivariate combination"

                alert_items.append({
                    "identifier": identifier_str,
                    "date": date_str,
                    "severity": severity,
                    "anomaly_score": float(row["anomaly_score"]),
                    "trigger_reasons": explanation,
                })

            insights = [
                f"Scanned {len(work_df):,} operational records: Flagged {n_anomalies} anomalous events ({anomaly_rate_pct}% outlier rate).",
                f"Model calibrated with an Isolation Forest ensemble (contamination factor: {self.contamination * 100:.1f}%).",
            ]

            if n_anomalies > 0:
                top_alert = alert_items[0]
                insights.append(
                    f"Highest Severity Outlier: Incident on {top_alert['date']} ({top_alert['identifier']}) triggered by: {top_alert['trigger_reasons']}."
                )

            return ModelResult(
                model_type=ModelType.ANOMALY_DETECTION,
                model_name="Isolation Forest Anomaly Radar",
                status="success",
                metrics={
                    "total_records_scanned": len(work_df),
                    "anomalies_detected": n_anomalies,
                    "anomaly_rate_pct": anomaly_rate_pct,
                    "features_evaluated": len(feature_cols),
                },
                insights=insights,
                quantitative_findings={
                    "total_anomalies": n_anomalies,
                    "anomaly_rate": anomaly_rate_pct,
                    "most_severe_score": float(anomalies_df["anomaly_score"].max()) if not anomalies_df.empty else 0.0,
                },
                tables={
                    "anomaly_alerts": alert_items,
                },
                charts_data={
                    "anomaly_counts": [
                        {"status": "Normal", "count": len(work_df) - n_anomalies},
                        {"status": "Anomalies", "count": n_anomalies},
                    ],

        return None
