"""
Conversational Natural Language Query Agent for SMB Business Owners.
Allows non-technical users to query their business dataset in plain English
and receives instant quantitative calculations and answers.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import re

from data_intelligence.agents.base import BaseAgent
from data_intelligence.types import DatasetProfile


class ConversationalQueryAgent(BaseAgent):
    """
    Translates business questions into Pandas analytical queries
    and synthesizes natural language answers with data tables.
    """

    def __init__(self, profile: Optional[DatasetProfile] = None):
        super().__init__(
            name="Conversational Query Agent",
            role="Answer ad-hoc business questions and calculate custom metrics from data",
        )
        self.profile = profile

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Base runner protocol."""
        return context

    def answer(self, question: str, df: pd.DataFrame, profile: Optional[DatasetProfile] = None) -> Dict[str, Any]:
        """
        Parse and answer a user question against the given dataframe.
        """
        active_profile = profile or self.profile
        q_low = question.lower().strip()

        # Target and numerical columns
        target_col = active_profile.detected_target_col if active_profile else None
        if not target_col:
            for c in df.columns:
                if any(k in c.lower() for k in ["revenue", "sales", "mrr", "amount", "total"]):
                    target_col = c
                    break

        cat_col = active_profile.detected_category_col if active_profile else None
        cust_col = active_profile.detected_customer_id_col if active_profile else None
        dt_col = active_profile.detected_datetime_col if active_profile else None

        # 1. Total Revenue / Gross Sales
        if any(w in q_low for w in ["total revenue", "total sales", "how much revenue", "gross sales", "total turnover"]):
            if target_col and target_col in df.columns:
                total_val = float(df[target_col].sum())
                avg_val = float(df[target_col].mean())
                return {
                    "question": question,
                    "answer": f"Total cumulative {target_col.replace('_', ' ')} across all records is **${total_val:,.2f}** (with an average transaction value of **${avg_val:,.2f}**).",
                    "metric_value": round(total_val, 2),
                    "table_data": None,
                }

        # 2. Total Customers / Accounts
        if any(w in q_low for w in ["how many customers", "customer count", "total users", "how many accounts", "total clients"]):
            if cust_col and cust_col in df.columns:
                n_unique = int(df[cust_col].nunique())
                return {
                    "question": question,
                    "answer": f"There are **{n_unique:,} unique customers** recorded in your dataset across {len(df):,} total transactions.",
                    "metric_value": n_unique,
                    "table_data": None,
                }
            return {
                "question": question,
                "answer": f"The dataset contains **{len(df):,} total records**.",
                "metric_value": len(df),
                "table_data": None,
            }

        # 3. Best / Top Customers
        if any(w in q_low for w in ["top customer", "best customer", "highest spender", "who spent the most", "top 5 customer", "top 10 customer"]):
            if cust_col and cust_col in df.columns and target_col and target_col in df.columns:
                top_c = (
                    df.groupby(cust_col)[target_col]
                    .sum()
                    .reset_index()
                    .sort_values(by=target_col, ascending=False)
                    .head(10)
                )
                top_one = top_c.iloc[0]
                table_list = top_c.to_dict(orient="records")
                return {
                    "question": question,
                    "answer": f"Your highest-spending customer is **{top_one[cust_col]}** with a cumulative spend of **${top_one[target_col]:,.2f}**. Here are your top accounts:",
                    "metric_value": round(float(top_one[target_col]), 2),
                    "table_data": table_list,
                }

        # 4. Top Product Categories / Segments
        if any(w in q_low for w in ["top category", "best product", "category breakdown", "sales by category", "most popular category", "highest revenue category"]):
            if cat_col and cat_col in df.columns and target_col and target_col in df.columns:
                cat_summary = (
                    df.groupby(cat_col)[target_col]
                    .agg(["sum", "count", "mean"])
                    .reset_index()
                    .rename(columns={"sum": "total_revenue", "count": "order_count", "mean": "avg_ticket"})
                    .sort_values(by="total_revenue", ascending=False)
                )
                top_cat = cat_summary.iloc[0]
                table_list = cat_summary.round(2).to_dict(orient="records")
                return {
                    "question": question,
                    "answer": f"The top performing category is **{top_cat[cat_col]}**, generating **${top_cat['total_revenue']:,.2f}** ({top_cat['order_count']:,} orders).",
                    "metric_value": round(float(top_cat["total_revenue"]), 2),
                    "table_data": table_list,
                }

        # 5. Churn Rate
        if any(w in q_low for w in ["churn rate", "how many churn", "churn percentage", "cancellation rate"]):
            churn_cols = [c for c in df.columns if any(k in c.lower() for k in ["churn", "churned", "is_churn"])]
            if churn_cols:
                c_col = churn_cols[0]
                rate = float(df[c_col].mean()) * 100
                count = int(df[c_col].sum())
                return {
                    "question": question,
                    "answer": f"Your current churn rate is **{rate:.1f}%** ({count:,} churned accounts out of {len(df):,} total).",
                    "metric_value": round(rate, 2),
                    "table_data": None,
                }

        # 6. Average Order / Check Size / Profit Margin
        if any(w in q_low for w in ["average order", "aov", "average check", "avg ticket", "average basket"]):
            if target_col and target_col in df.columns:
                aov = float(df[target_col].mean())
                return {
                    "question": question,
                    "answer": f"The average transaction / order value (AOV) is **${aov:,.2f}**.",
                    "metric_value": round(aov, 2),
                    "table_data": None,
                }

        if any(w in q_low for w in ["profit margin", "average profit", "profitability", "net profit"]):
            profit_cols = [c for c in df.columns if any(k in c.lower() for k in ["profit", "net_profit", "margin"])]
            if profit_cols:
                p_col = profit_cols[0]
                tot_p = float(df[p_col].sum())
                if target_col and target_col in df.columns:
                    margin_pct = (tot_p / df[target_col].sum()) * 100
                    return {
                        "question": question,
                        "answer": f"Cumulative profit is **${tot_p:,.2f}**, representing an overall profit margin of **{margin_pct:.1f}%**.",
                        "metric_value": round(tot_p, 2),
                        "table_data": None,
                    }

        # 7. Day of Week / Temporal Questions
        if any(w in q_low for w in ["day of week", "best day", "weekend vs weekday", "busiest day"]):
            if "day_of_week" in df.columns and target_col and target_col in df.columns:
                dow_summary = (
                    df.groupby("day_of_week")[target_col]
                    .mean()
                    .reset_index()
                    .sort_values(by=target_col, ascending=False)
                )
                busiest = dow_summary.iloc[0]
                return {
                    "question": question,
                    "answer": f"Your highest revenue velocity occurs on **{busiest['day_of_week']}** with an average of **${busiest[target_col]:,.2f}** per period.",
                    "metric_value": round(float(busiest[target_col]), 2),
                    "table_data": dow_summary.round(2).to_dict(orient="records"),
                }

        # 8. General numerical summary fallback
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        summary_stats = df[num_cols[:5]].describe().transpose().round(2).reset_index()
        return {
            "question": question,
            "answer": f"Here is the numerical statistical summary for your key attributes ({', '.join(num_cols[:5])}):",
            "metric_value": None,
            "table_data": summary_stats.to_dict(orient="records"),
        }
