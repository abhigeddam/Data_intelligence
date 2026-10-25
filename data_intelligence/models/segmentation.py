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

        return None
