"""
Customer RFM (Recency, Frequency, Monetary) and K-Means Behavioral Segmentation Engine.
Identifies customer tiers, revenue concentration, at-risk accounts, and targeted marketing cohorts.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from data_intelligence.types import ModelType, ModelResult, DatasetProfile


class CustomerSegmentationEngine:
    """
    Automated customer segmentation engine implementing RFM analysis
    and unsupervised clustering for SMBs.
    """

    def __init__(self, profile: DatasetProfile, max_clusters: int = 5):
        self.profile = profile
        self.max_clusters = max_clusters

    def run(
        self,
        df: pd.DataFrame,
        customer_col: Optional[str] = None,
        date_col: Optional[str] = None,
        monetary_col: Optional[str] = None,
    ) -> ModelResult:
        """
        Execute RFM segmentation and K-Means clustering.
        """
        start_time = time.time()
        c_col = customer_col or self.profile.detected_customer_id_col
        d_col = date_col or self.profile.detected_datetime_col
        m_col = monetary_col or self.profile.detected_target_col

        if not c_col or c_col not in df.columns:
            return ModelResult(
                model_type=ModelType.CUSTOMER_SEGMENTATION,
                model_name="RFM & K-Means Behavioral Clustering",
                status="skipped",
                insights=["Segmentation skipped: No customer identifier column detected."],
                error_message="Customer ID column not found.",
                execution_time_seconds=round(time.time() - start_time, 2),
            )

        try:
            work_df = df.copy()

            # If transaction-level data with dates and monetary values
            if d_col and d_col in work_df.columns and m_col and m_col in work_df.columns:
                work_df[d_col] = pd.to_datetime(work_df[d_col], errors="coerce")
                work_df = work_df.dropna(subset=[c_col, d_col, m_col])
                reference_date = work_df[d_col].max() + pd.Timedelta(days=1)

                rfm = work_df.groupby(c_col).agg(
                    recency=(d_col, lambda x: (reference_date - x.max()).days),
                    frequency=(c_col, "count"),
                    monetary=(m_col, "sum"),
                ).reset_index()
            else:
                # If already an aggregated customer-level table (e.g. SaaS with tenure, mrr)
                num_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
                if len(num_cols) < 2:
                    return ModelResult(
                        model_type=ModelType.CUSTOMER_SEGMENTATION,
                        model_name="RFM & K-Means Behavioral Clustering",
                        status="skipped",
                        insights=["Insufficient numerical metrics to formulate behavioral clustering."],
                        execution_time_seconds=round(time.time() - start_time, 2),
                    )
                rfm = work_df[[c_col] + num_cols[:3]].dropna().copy()
                col_names = ["customer_id", "recency", "frequency", "monetary"]
                # Map available columns
                rfm.columns = [c_col, "recency", "frequency"] if len(num_cols) == 2 else [c_col, "recency", "frequency", "monetary"]
                if "monetary" not in rfm.columns:
                    rfm["monetary"] = rfm["frequency"] * 100.0

            if len(rfm) < 10:
                return ModelResult(
                    model_type=ModelType.CUSTOMER_SEGMENTATION,
                    model_name="RFM & K-Means Behavioral Clustering",
                    status="warning",
                    insights=["Fewer than 10 unique customers available. Segmentation requires a larger customer base."],
                    execution_time_seconds=round(time.time() - start_time, 2),
                )

            # Prevent zero or negative values in log transform
            rfm["recency"] = np.maximum(rfm["recency"], 1)
            rfm["frequency"] = np.maximum(rfm["frequency"], 1)
            rfm["monetary"] = np.maximum(rfm["monetary"], 0.01)

            # Log transform to manage skewness
            log_rfm = np.log1p(rfm[["recency", "frequency", "monetary"]])
            scaler = StandardScaler()
            scaled_rfm = scaler.fit_transform(log_rfm)

            # Select optimal k based on silhouette score
            k_candidates = [3, 4, 5]
            best_k = 4
            best_score = -1.0
            best_labels = None

            for k in k_candidates:
                if len(rfm) <= k:
                    continue
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(scaled_rfm)
                score = silhouette_score(scaled_rfm, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels

            if best_labels is None:
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                best_labels = kmeans.fit_predict(scaled_rfm)
                best_k = 3
                best_score = 0.35

            rfm["cluster"] = best_labels

            # Cluster aggregation to assign human persona tags
            cluster_stats = rfm.groupby("cluster").agg(
                customer_count=(c_col, "count"),
                avg_recency=("recency", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"),
                total_monetary=("monetary", "sum"),
            ).reset_index()

            total_revenue = rfm["monetary"].sum()
            total_cust = len(rfm)
            cluster_stats["revenue_pct"] = (cluster_stats["total_monetary"] / max(0.01, total_revenue)) * 100
            cluster_stats["customer_pct"] = (cluster_stats["customer_count"] / total_cust) * 100

            # Rank clusters by monetary and recency to label personas
            def assign_persona(row: pd.Series) -> str:
                # Rank logic
                if row["avg_monetary"] >= cluster_stats["avg_monetary"].quantile(0.70):
                    if row["avg_recency"] <= cluster_stats["avg_recency"].median():
                        return "Champions & VIPs"
                    return "At-Risk High-Value"
                elif row["avg_frequency"] >= cluster_stats["avg_frequency"].median():
                    return "Loyal Repeat Customers"

        return None
