"""
Tests for DataProfiler and DataCleaner.
"""

import pytest
import pandas as pd
import numpy as np

from data_intelligence.core.profiler import DataProfiler
from data_intelligence.core.cleaner import DataCleaner
from data_intelligence.types import BusinessType, ColumnSemanticType


@pytest.fixture
def sample_raw_dataframe():
    return pd.DataFrame({
        "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-4", "ORD-4"],  # Contains duplicate
        "order_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-04"],
        "customer_id": ["C101", "C102", "C101", "C103", "C103"],
        "product_category": ["Electronics", "Apparel", "Electronics", "Apparel", "Apparel"],
        "gross_revenue": [120.5, 45.0, np.nan, 85.0, 85.0],  # Contains NaN
        "unit_price": [120.5, 45.0, 60.0, 85.0, 85.0],
        "quantity": [1, 1, 2, 1, 1],
        "constant_col": [1, 1, 1, 1, 1],  # Zero variance
    })


def test_data_profiler_semantics(sample_raw_dataframe):
    profiler = DataProfiler()
    profile = profiler.profile(sample_raw_dataframe, dataset_name="test_order_data")

    assert profile.shape == [5, 8]
    assert profile.detected_datetime_col == "order_date"
    assert profile.detected_target_col == "gross_revenue"
    assert profile.detected_customer_id_col == "customer_id"
    assert profile.detected_category_col == "product_category"
    assert profile.detected_quantity_col == "quantity"
    assert profile.columns["constant_col"].is_constant is True
    assert profile.data_quality.total_duplicates == 1
    assert "gross_revenue" in profile.data_quality.columns_with_missing


def test_data_cleaner(sample_raw_dataframe):
    profiler = DataProfiler()
    profile = profiler.profile(sample_raw_dataframe)

    cleaner = DataCleaner(profile)
    cleaned_df, audit = cleaner.clean(sample_raw_dataframe)

    assert audit["duplicates_removed"] == 1
    assert "constant_col" in audit["dropped_columns"]
    assert "constant_col" not in cleaned_df.columns
    assert cleaned_df["gross_revenue"].isna().sum() == 0
    assert len(cleaned_df) == 4
