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
                audit_log["dropped_columns"].append(col_name)

        # 3. Parse Datetime columns
        if self.profile.detected_datetime_col and self.profile.detected_datetime_col in cleaned.columns:
            dt_col = self.profile.detected_datetime_col
            try:
                cleaned[dt_col] = pd.to_datetime(cleaned[dt_col], errors="coerce")
                audit_log["datetime_parsed"].append(dt_col)
            except Exception:
                pass

        # 4. Handle Missing Values
        for col in cleaned.columns:
            n_missing = int(cleaned[col].isna().sum())
            if n_missing == 0:
                continue

            if pd.api.types.is_numeric_dtype(cleaned[col]):
                # Median imputation for robustness against skew
                median_val = cleaned[col].median()
                cleaned[col] = cleaned[col].fillna(median_val)
                audit_log["imputed_columns"][col] = {
                    "method": "median",
                    "value": float(round(median_val, 3)) if pd.notna(median_val) else 0.0,
                    "imputed_count": n_missing,
                }
            elif pd.api.types.is_datetime64_any_dtype(cleaned[col]):
                # Forward fill then backward fill for time series
                cleaned[col] = cleaned[col].ffill().bfill()
                audit_log["imputed_columns"][col] = {
                    "method": "ffill_bfill",
                    "imputed_count": n_missing,
                }
            else:
                # Mode or "Unknown"
                mode_val = cleaned[col].mode()
                val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
                cleaned[col] = cleaned[col].fillna(val)
                audit_log["imputed_columns"][col] = {
                    "method": "mode",
                    "value": str(val),
                    "imputed_count": n_missing,
                }

        audit_log["final_rows"] = len(cleaned)
        audit_log["final_columns"] = len(cleaned.columns)
        return cleaned, audit_log
