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
