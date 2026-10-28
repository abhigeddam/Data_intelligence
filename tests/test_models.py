"""
Tests for the 5 analytical model engines.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_intelligence.core.profiler import DataProfiler
from data_intelligence.models.forecaster import TimeSeriesForecaster
from data_intelligence.models.segmentation import CustomerSegmentationEngine
from data_intelligence.models.churn import ChurnPredictionEngine
from data_intelligence.models.elasticity import PriceElasticityEngine
from data_intelligence.models.anomaly import AnomalyDetectionEngine


@pytest.fixture
def mock_retail_dataset():
    np.random.seed(42)
    start = datetime(2025, 1, 1)
    dates = [start + timedelta(days=i) for i in range(120)]
    records = []
    for i, d in enumerate(dates):
        u_price = 50.0 + (i % 10) * 5.0
        qty = 1 + (i % 4)
        records.append({
            "order_date": d.strftime("%Y-%m-%d"),
            "customer_id": f"CUST-{100 + (i % 25)}",
            "product_category": "Apparel" if i % 2 == 0 else "Electronics",
            "unit_price": u_price,
            "quantity": qty,
            "gross_revenue": u_price * qty,
            "cost": 30.0 * qty,
            "churned": 1 if u_price >= 75.0 else 0,
        })
    return pd.DataFrame(records)


def test_forecaster_engine(mock_retail_dataset):
    profiler = DataProfiler()
    profile = profiler.profile(mock_retail_dataset)

    forecaster = TimeSeriesForecaster(profile, horizon_days=14)
    res = forecaster.run(mock_retail_dataset)

    assert res.status == "success"
    assert "projected_total" in res.quantitative_findings
    assert len(res.tables["forecast_summary"]) == 14
    assert res.metrics["MAPE_pct"] >= 0


def test_customer_segmentation_engine(mock_retail_dataset):
    profiler = DataProfiler()
    profile = profiler.profile(mock_retail_dataset)

    segmenter = CustomerSegmentationEngine(profile)
    res = segmenter.run(mock_retail_dataset)

    assert res.status == "success"
    assert res.metrics["total_customers"] == 25
    assert len(res.tables["segment_summary"]) > 0


def test_churn_engine(mock_retail_dataset):
    profiler = DataProfiler()
    profile = profiler.profile(mock_retail_dataset)

    churner = ChurnPredictionEngine(profile)
    res = churner.run(mock_retail_dataset)

    assert res.status == "success"
    assert res.metrics["roc_auc"] >= 0.5
    assert len(res.tables["top_churn_drivers"]) > 0


def test_elasticity_engine(mock_retail_dataset):
    profiler = DataProfiler()
    profile = profiler.profile(mock_retail_dataset)

    elasticity = PriceElasticityEngine(profile)
    res = elasticity.run(mock_retail_dataset)

    assert res.status == "success"
    assert "overall_elasticity" in res.metrics


def test_anomaly_engine(mock_retail_dataset):
    profiler = DataProfiler()
    profile = profiler.profile(mock_retail_dataset)

    anomaly = AnomalyDetectionEngine(profile, contamination=0.05)
    res = anomaly.run(mock_retail_dataset)

    assert res.status == "success"
    assert res.metrics["anomalies_detected"] > 0
