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
def get_orchestrator():
    return AgentOrchestrator()


@st.cache_resource
def get_query_agent():
    return ConversationalQueryAgent()


orchestrator = get_orchestrator()
query_agent = get_query_agent()

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("OmniInsight AI")
    st.caption("Autonomous Multi-Model SMB Intelligence")
    st.markdown("---")

    data_source_mode = st.radio(
        "Choose Data Source:",
        [
            "🛒 Retail & E-Commerce Demo",
            "💻 SaaS Customer Retention Demo",
            "🍽️ Restaurant Operations Demo",
            "📂 Upload My Custom CSV / Excel",
        ],
    )

    business_type_override = st.selectbox(
        "Business Domain:",
        ["Auto-Detect", "Retail", "SaaS", "Restaurant", "General"],
        index=0,
    )

    uploaded_file = None
    if "Upload" in data_source_mode:
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

    st.markdown("---")
    run_btn = st.button("🚀 Run Multi-Agent Audit", type="primary", use_container_width=True)


# Data Ingestion Logic
df_active = None
active_name = "sample_data"
inferred_type = None

if "Retail" in data_source_mode:
    sample_file = settings.samples_dir / "retail_ecommerce_sales.csv"
    if sample_file.exists():
        df_active = pd.read_csv(sample_file)
        active_name = "Retail E-Commerce Transactions"
        inferred_type = BusinessType.RETAIL
elif "SaaS" in data_source_mode:
    sample_file = settings.samples_dir / "saas_customer_retention.csv"
    if sample_file.exists():
        df_active = pd.read_csv(sample_file)
        active_name = "SaaS Subscriptions & Churn"
        inferred_type = BusinessType.SAAS
elif "Restaurant" in data_source_mode:
    sample_file = settings.samples_dir / "restaurant_operations.csv"
    if sample_file.exists():
        df_active = pd.read_csv(sample_file)
        active_name = "Restaurant Operations & Covers"
        inferred_type = BusinessType.RESTAURANT
elif uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_active = pd.read_csv(uploaded_file)
        else:
            df_active = pd.read_excel(uploaded_file)
        active_name = uploaded_file.name.rsplit(".", 1)[0]
    except Exception as e:
        st.error(f"Failed to read uploaded file: {str(e)}")
