"""
OmniInsight AI: Autonomous Multi-Model Data Intelligence Platform for SMBs.
Interactive Streamlit Web Application.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import io

from data_intelligence.agents.orchestrator import AgentOrchestrator
from data_intelligence.agents.query_agent import ConversationalQueryAgent
from data_intelligence.types import BusinessType, ModelType
from data_intelligence.config import settings
from data_intelligence.reporting.markdown_report import generate_markdown_report
from data_intelligence.ui.visualizations import (
    plot_time_series_forecast,
    plot_seasonality_dow,
    plot_customer_segments,
    plot_churn_drivers,
    plot_price_elasticity_scatter,
)

# Page Configuration
st.set_page_config(
    page_title="OmniInsight AI | SMB Data Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
