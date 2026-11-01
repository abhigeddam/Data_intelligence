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

        return {}
