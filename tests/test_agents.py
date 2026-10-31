"""
Tests for individual agents and conversational query agent.
"""

import pytest
import pandas as pd
from data_intelligence.core.profiler import DataProfiler
from data_intelligence.agents.hypothesis_agent import BusinessStrategyHypothesisAgent
from data_intelligence.agents.executive_agent import ExecutiveNarrativeAgent
from data_intelligence.agents.query_agent import ConversationalQueryAgent
from data_intelligence.types import BusinessType, ModelResult, ModelType


def test_hypothesis_agent():
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "customer_id": ["C1", "C2"],
        "total_revenue": [100.0, 200.0],
        "quantity": [1, 2],
        "churned": [0, 1],
    })
    profiler = DataProfiler(business_type=BusinessType.RETAIL)
    profile = profiler.profile(df)

    agent = BusinessStrategyHypothesisAgent()
    ctx = {"dataset_profile": profile}
    ctx = agent.run(ctx)

    assert "hypotheses" in ctx
    assert len(ctx["hypotheses"]) >= 4


def test_conversational_query_agent():
    df = pd.DataFrame({
        "order_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "customer_id": ["CUST-A", "CUST-B", "CUST-A"],
        "gross_revenue": [500.0, 300.0, 250.0],
        "product_category": ["Electronics", "Apparel", "Electronics"],
        "churned": [0, 1, 0],
    })
    profiler = DataProfiler()
    profile = profiler.profile(df)

    qa = ConversationalQueryAgent(profile=profile)

    res1 = qa.answer("What is our total revenue?", df=df)
    assert "1,050.00" in res1["answer"]

    res2 = qa.answer("Who is our top customer?", df=df)
    assert "CUST-A" in res2["answer"]

    res3 = qa.answer("What is the churn rate?", df=df)
    assert "33.3%" in res3["answer"]
