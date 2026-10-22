"""
Automated Data Profiler and Semantic Detector.
Examines tabular business data, infers column roles, audits data quality,
and derives a comprehensive dataset profile for downstream multi-agent modeling.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from data_intelligence.types import (
    BusinessType,
    ColumnSemanticType,
    ColumnProfile,
    DataQualityScore,
    DatasetProfile,
)


class DataProfiler:
    """
    Introspects business tabular data, detects semantic roles of columns,
    evaluates data hygiene, and recommends analytics targets.
    """

    DATETIME_KEYWORDS = ["date", "time", "timestamp", "created_at", "signup", "period", "day"]
    REVENUE_KEYWORDS = ["revenue", "sales", "gross", "mrr", "arr", "check", "amount", "total"]
    COST_KEYWORDS = ["cost", "cogs", "expense", "labor", "spend", "overhead"]
    PROFIT_KEYWORDS = ["profit", "margin", "net", "ebitda"]
    CUSTOMER_KEYWORDS = ["customer", "cust", "user", "account", "client", "buyer"]
    CHURN_KEYWORDS = ["churn", "churned", "canceled", "cancelled", "attrition", "left", "inactive"]
    QUANTITY_KEYWORDS = ["qty", "quantity", "volume", "units", "covers", "orders", "count"]
    CATEGORY_KEYWORDS = ["category", "dept", "tier", "plan", "channel", "group", "type", "segment"]

    def __init__(self, business_type: Optional[BusinessType] = None):
        self.business_type = business_type

    def profile(self, df: pd.DataFrame, dataset_name: str = "dataset") -> DatasetProfile:
        """
        Execute full profiling pipeline on a pandas DataFrame.
        """
        df_clean = df.copy()
        n_rows, n_cols = df_clean.shape
        memory_mb = round(df_clean.memory_usage(deep=True).sum() / (1024 * 1024), 3)

        # 1. Infer business type if not provided
        b_type = self.business_type or self._infer_business_type(df_clean)

        # 2. Profile each column
        columns_profile: Dict[str, ColumnProfile] = {}
        for col_name in df_clean.columns:
            columns_profile[col_name] = self._profile_column(df_clean[col_name])

        # 3. Calculate data quality score
        quality_score = self._compute_quality_score(df_clean, columns_profile)

        # 4. Detect semantic roles
        dt_col = self._detect_datetime_column(df_clean, columns_profile)
        target_col = self._detect_target_column(df_clean, columns_profile)
        cust_col = self._detect_customer_column(df_clean, columns_profile)
        cat_col = self._detect_category_column(df_clean, columns_profile)
        qty_col = self._detect_quantity_column(df_clean, columns_profile)
        cost_col = self._detect_cost_column(df_clean, columns_profile)

        summary_text = (
            f"Dataset '{dataset_name}' contains {n_rows:,} records and {n_cols} attributes "
            f"classified under {b_type.value.upper()} domain. Overall data hygiene rating is "
            f"{quality_score.overall_score:.1f}/100. Primary metric detected: '{target_col or 'N/A'}'."
        )

        return DatasetProfile(
            business_type=b_type,
            shape=[n_rows, n_cols],
            memory_usage_mb=memory_mb,
            columns=columns_profile,
            data_quality=quality_score,
            detected_datetime_col=dt_col,
            detected_target_col=target_col,
            detected_customer_id_col=cust_col,
            detected_category_col=cat_col,
            detected_quantity_col=qty_col,
            detected_cost_col=cost_col,
            summary_text=summary_text,
        )

    def _infer_business_type(self, df: pd.DataFrame) -> BusinessType:
        """
        Infer business category from column nomenclature and presence of key attributes.
        """
        cols_lower = [str(c).lower() for c in df.columns]

        if any(w in cols_lower for w in ["mrr", "arr", "seats_purchased", "nps_score", "plan_tier"]):
            return BusinessType.SAAS
        if any(w in cols_lower for w in ["covers_total", "dine_in_sales", "delivery_sales", "table_turnover_rate"]):
            return BusinessType.RESTAURANT
        if any(w in cols_lower for w in ["product_category", "unit_price", "discount_rate", "order_id"]):
            return BusinessType.RETAIL
        if any("ecommerce" in c or "cart" in c or "sku" in c for c in cols_lower):
            return BusinessType.ECOMMERCE
        return BusinessType.GENERAL

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        """
        Deep inspection of a single pandas Series.
        """
        col_name = str(series.name)
        total_count = len(series)
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / total_count) * 100, 2) if total_count > 0 else 0.0

        non_na = series.dropna()
        unique_count = int(non_na.nunique())
        unique_pct = round((unique_count / total_count) * 100, 2) if total_count > 0 else 0.0
        is_constant = unique_count <= 1 and total_count > 0
        is_unique = unique_count == total_count and total_count > 0

        # Semantic Type Inference
        semantic_type = self._infer_semantic_type(col_name, series, non_na, unique_count, total_count)

        # Statistics computation
        stats: Dict[str, Any] = {}
        sample_vals = non_na.head(5).tolist()

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            if len(non_na) > 0:
                stats = {
                    "mean": float(round(non_na.mean(), 2)),
                    "std": float(round(non_na.std(), 2)) if len(non_na) > 1 else 0.0,
                    "min": float(round(non_na.min(), 2)),
                    "25%": float(round(non_na.quantile(0.25), 2)),
                    "50%": float(round(non_na.median(), 2)),
                    "75%": float(round(non_na.quantile(0.75), 2)),
                    "max": float(round(non_na.max(), 2)),
                }
        elif pd.api.types.is_datetime64_any_dtype(series):
            if len(non_na) > 0:
                stats = {
                    "min_date": str(non_na.min()),
                    "max_date": str(non_na.max()),
                    "date_span_days": int((non_na.max() - non_na.min()).days),
                }
        else:
            # Categorical / string distribution
            if len(non_na) > 0:
                top_counts = non_na.value_counts().head(5).to_dict()
                stats = {
                    "top_categories": {str(k): int(v) for k, v in top_counts.items()}
                }

        return ColumnProfile(
            name=col_name,
            semantic_type=semantic_type,
            dtype=str(series.dtype),
            total_count=total_count,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            unique_pct=unique_pct,
            is_constant=is_constant,
            is_unique=is_unique,
            stats=stats,
            sample_values=sample_vals,
        )

    def _infer_semantic_type(
        self,
        name: str,
        series: pd.Series,
        non_na: pd.Series,
        unique_count: int,
        total_count: int,
    ) -> ColumnSemanticType:
        """
        Infer semantic data type based on naming patterns and distribution values.
        """
        name_lower = name.lower()

        # 1. Identifier
        if (unique_count / max(1, total_count) > 0.85) and any(
            id_kw in name_lower for id_kw in ["id", "uuid", "hash", "key", "account_name"]
        ):
            return ColumnSemanticType.IDENTIFIER

        # 2. Boolean
        if pd.api.types.is_bool_dtype(series) or (
            pd.api.types.is_numeric_dtype(series)
            and set(non_na.unique()).issubset({0, 1})
            and len(non_na.unique()) <= 2
        ):
            return ColumnSemanticType.BOOLEAN

        # 3. Datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnSemanticType.DATETIME

        # Check if string column is actually a datetime
        if series.dtype == "object" and len(non_na) > 0:
            if any(kw in name_lower for kw in self.DATETIME_KEYWORDS):
                try:
                    pd.to_datetime(non_na.head(20), errors="raise")
                    return ColumnSemanticType.DATETIME
                except Exception:
                    pass

        # 4. Numeric continuous vs discrete
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_float_dtype(series) or unique_count > 25:
                return ColumnSemanticType.NUMERIC_CONTINUOUS
            return ColumnSemanticType.NUMERIC_DISCRETE

        # 5. Categorical vs free text
        if unique_count <= 50 or (unique_count / max(1, total_count) < 0.2):
            return ColumnSemanticType.CATEGORICAL

        return ColumnSemanticType.TEXT

    def _compute_quality_score(
        self,
        df: pd.DataFrame,
        cols: Dict[str, ColumnProfile],
    ) -> DataQualityScore:
        """
        Calculate an interpretable 0-100 data hygiene score with diagnostic alerts.
        """
        n_rows, n_cols = df.shape
        if n_rows == 0:
            return DataQualityScore(
                overall_score=0.0,
                completeness_score=0.0,
                uniqueness_score=0.0,
                consistency_score=0.0,
                total_rows=0,
                total_columns=n_cols,
                total_duplicates=0,
                critical_issues=["Dataset is empty."],
            )

        # Duplicate check
        n_dups = int(df.duplicated().sum())
        uniqueness_score = max(0.0, 100.0 - (n_dups / n_rows * 100.0 * 2.0))

        # Missingness check
        total_cells = n_rows * n_cols
        total_missing = sum(c.missing_count for c in cols.values())
        missing_rate = total_missing / total_cells if total_cells > 0 else 0.0
        completeness_score = max(0.0, 100.0 - (missing_rate * 100.0 * 3.0))

        # Consistency check (constant columns, broken types)
        constant_cols = [c.name for c in cols.values() if c.is_constant]
        consistency_penalty = len(constant_cols) * 8.0
        consistency_score = max(0.0, 100.0 - consistency_penalty)

        overall = round(
            0.45 * completeness_score + 0.35 * uniqueness_score + 0.20 * consistency_score, 1
        )

        critical_issues: List[str] = []
        warnings: List[str] = []
        recommendations: List[str] = []
        cols_with_missing = [c.name for c in cols.values() if c.missing_count > 0]

        if n_dups > 0:
            pct_dup = round(n_dups / n_rows * 100, 1)
            warnings.append(f"Found {n_dups:,} duplicate rows ({pct_dup}% of total).")
            recommendations.append("Deduplicate dataset before building predictive models.")

        if missing_rate > 0.15:
            critical_issues.append(f"High missingness: {round(missing_rate * 100, 1)}% of total data cells are null.")
            recommendations.append("Apply automated imputation for critical numerical features.")
        elif missing_rate > 0.02:
            warnings.append(f"Moderate missingness detected ({round(missing_rate * 100, 1)}% null values).")

        if constant_cols:
            warnings.append(f"Constant columns with zero variance detected: {', '.join(constant_cols)}")
            recommendations.append(f"Drop uninformative columns: {', '.join(constant_cols)}")

        if overall >= 85.0 and not critical_issues:
            recommendations.append("Data health is robust and ready for multi-model synthesis.")

        return DataQualityScore(

        return None
