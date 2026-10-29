"""
Data Intake, Profiling, and Sanitization Agent.
"""

from typing import Dict, Any, Optional
import pandas as pd

from data_intelligence.agents.base import BaseAgent
from data_intelligence.core.profiler import DataProfiler
from data_intelligence.core.cleaner import DataCleaner
from data_intelligence.types import BusinessType, DatasetProfile


class DataProfilerAgent(BaseAgent):
    """
    Ingests raw customer data, conducts statistical profiling,
    identifies semantic column roles, and sanitizes data for analysis.
    """

    def __init__(self, business_type: Optional[BusinessType] = None):
        super().__init__(
            name="Data Intake & Profiler Agent",
            role="Inspect schema, calculate hygiene rating, detect semantic roles, and sanitize data",
        )
        self.business_type = business_type

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raw_df: pd.DataFrame = context.get("raw_df")
        if raw_df is None or raw_df.empty:
            raise ValueError("Input dataframe is empty or missing in workflow context.")

        dataset_name = context.get("dataset_name", "business_data")
        b_type = self.business_type or context.get("business_type")

        # 1. Profile data
        profiler = DataProfiler(business_type=b_type)
        profile = profiler.profile(raw_df, dataset_name=dataset_name)

        # 2. Clean and sanitize data
        cleaner = DataCleaner(profile)
        cleaned_df, cleaning_audit = cleaner.clean(raw_df)

        # Update context
        context["dataset_profile"] = profile
        context["cleaned_df"] = cleaned_df
        context["cleaning_audit"] = cleaning_audit
        context["business_type"] = profile.business_type

        return context
