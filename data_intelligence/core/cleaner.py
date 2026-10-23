"""
Automated Data Cleaner and Sanitizer for SMB Datasets.
Applies robust cleaning pipelines without manual user intervention.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from data_intelligence.types import DatasetProfile, ColumnSemanticType


class DataCleaner:
    """
    Cleans, deduplicates, and prepares raw SMB data for modeling.
    """

    def __init__(self, profile: DatasetProfile):
        self.profile = profile

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute full sanitation pipeline on DataFrame.
        Returns:
            (cleaned_df, transformation_audit_log)
        """
        cleaned = df.copy()
        audit_log: Dict[str, Any] = {
            "initial_rows": len(df),
            "initial_columns": len(df.columns),
            "duplicates_removed": 0,
            "imputed_columns": {},
            "datetime_parsed": [],
            "dropped_columns": [],
        }

        # 1. Deduplicate
        initial_count = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        audit_log["duplicates_removed"] = initial_count - len(cleaned)

        # 2. Drop constant columns (zero variance)
        for col_name, col_meta in self.profile.columns.items():
            if col_meta.is_constant and col_name in cleaned.columns:
                cleaned = cleaned.drop(columns=[col_name])

        return cleaned, audit_log
