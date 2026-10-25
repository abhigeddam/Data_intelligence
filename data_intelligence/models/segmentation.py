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
                elif row["avg_recency"] <= cluster_stats["avg_recency"].quantile(0.40):
                    return "Promising Newcomers"
                else:
                    return "Hibernating / Dormant"

            cluster_stats["persona"] = cluster_stats.apply(assign_persona, axis=1)

            # Map persona back to individual customer table
            persona_map = dict(zip(cluster_stats["cluster"], cluster_stats["persona"]))
            rfm["persona"] = rfm["cluster"].map(persona_map)

            # Find top revenue segment
            top_rev_segment = cluster_stats.sort_values(by="revenue_pct", ascending=False).iloc[0]
            at_risk_segment = cluster_stats[cluster_stats["persona"] == "At-Risk High-Value"]

            insights = [
                f"Identified {best_k} distinct behavioral segments with a Silhouette validation score of {best_score:.2f}.",
                f"Revenue Concentration: The '{top_rev_segment['persona']}' segment represents {top_rev_segment['customer_pct']:.1f}% of accounts but drives {top_rev_segment['revenue_pct']:.1f}% of total revenue (${top_rev_segment['total_monetary']:,.2f}).",
            ]

            if not at_risk_segment.empty:
                ar = at_risk_segment.iloc[0]
                insights.append(
                    f"Retention Alert: {int(ar['customer_count'])} accounts in '{ar['persona']}' have lapsed (avg recency {ar['avg_recency']:.0f} days) representing ${ar['total_monetary']:,.2f} in historical revenue."
                )

            # Construct clean output tables
            summary_table = []
            for _, row in cluster_stats.iterrows():
                summary_table.append({
                    "cluster_id": int(row["cluster"]),
                    "persona": str(row["persona"]),
                    "customer_count": int(row["customer_count"]),
                    "customer_pct": round(float(row["customer_pct"]), 1),
                    "avg_recency_days": round(float(row["avg_recency"]), 1),
                    "avg_orders_count": round(float(row["avg_frequency"]), 1),
                    "avg_spend_usd": round(float(row["avg_monetary"]), 2),
                    "total_revenue_usd": round(float(row["total_monetary"]), 2),
                    "revenue_share_pct": round(float(row["revenue_pct"]), 1),
                })

            sample_customers = rfm[[c_col, "recency", "frequency", "monetary", "persona"]].head(50).to_dict(orient="records")

            return ModelResult(
                model_type=ModelType.CUSTOMER_SEGMENTATION,
                model_name="RFM & K-Means Behavioral Clustering",
                status="success",
                metrics={
                    "total_customers": total_cust,
                    "optimal_clusters": best_k,
                    "silhouette_score": round(float(best_score), 3),
                    "total_analyzed_revenue": round(float(total_revenue), 2),
                },
                insights=insights,
                quantitative_findings={
                    "top_segment_name": str(top_rev_segment["persona"]),
                    "top_segment_rev_share": round(float(top_rev_segment["revenue_pct"]), 1),
                    "top_segment_count": int(top_rev_segment["customer_count"]),
                },
                tables={
                    "segment_summary": summary_table,
                    "customer_samples": sample_customers,
                },
                charts_data={
                    "segments": [s["persona"] for s in summary_table],
                    "revenue_shares": [s["revenue_share_pct"] for s in summary_table],
                    "customer_shares": [s["customer_pct"] for s in summary_table],
                    "recency_vs_monetary": [
                        {
                            "recency": float(r["recency"]),
                            "monetary": float(r["monetary"]),
                            "frequency": float(r["frequency"]),
                            "persona": str(r["persona"]),
                        }
                        for _, r in rfm.sample(n=min(300, len(rfm)), random_state=42).iterrows()
                    ],
                },
                execution_time_seconds=round(time.time() - start_time, 2),
            )
        except Exception as e:
            return ModelResult(
                model_type=ModelType.CUSTOMER_SEGMENTATION,
                model_name="RFM & K-Means Behavioral Clustering",
                status="error",
                error_message=str(e),
                insights=[f"Customer segmentation encountered an error: {str(e)}"],
                execution_time_seconds=round(time.time() - start_time, 2),
            )
