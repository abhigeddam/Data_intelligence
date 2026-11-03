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

# Header
st.markdown('<div class="main-header">⚡ OmniInsight: Autonomous SMB Data Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-agent AI platform orchestrating forecasting, customer RFM cohorts, churn risk radar, and econometric pricing models.</div>',
    unsafe_allow_html=True,
)

if df_active is None:
    st.info("👈 Select a sample benchmark dataset or upload your business data from the sidebar to begin.")
    st.stop()

# Auto-run or manually triggered
if "current_report" not in st.session_state or run_btn:
    with st.spinner("🤖 Autonomous AI Agent team analyzing dataset..."):
        b_type_arg = None
        if business_type_override != "Auto-Detect":
            b_type_arg = BusinessType(business_type_override.lower())
        else:
            b_type_arg = inferred_type

        report = orchestrator.run_from_dataframe(
            df=df_active,
            dataset_name=active_name,
            business_type=b_type_arg,
        )
        st.session_state["current_report"] = report
        st.session_state["active_df"] = df_active

report = st.session_state["current_report"]
df_active = st.session_state["active_df"]
p = report.dataset_profile
q = p.data_quality
ex = report.executive_summary
models = report.model_results

# Dashboard Tabs
tab_exec, tab_fc, tab_rfm, tab_churn, tab_price, tab_anom, tab_query = st.tabs([
    "🏢 Executive Brief",
    "📈 Demand Forecast",
    "👥 Customer RFM",
    "⚠️ Churn Radar",
    "🏷️ Pricing Power",
    "🛡️ Anomaly Radar",
    "💬 Ask AI Data Assistant",
])

# ----------------- TAB 1: EXECUTIVE BRIEF -----------------
with tab_exec:
    # KPI Highlights
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Health Score", f"{q.overall_score:.1f} / 100", delta="Audit Grade A" if q.overall_score >= 85 else "Hygiene Warnings")
    with col2:
        st.metric("Total Records", f"{p.shape[0]:,}", delta=f"{p.shape[1]} attributes")
    with col3:
        roi = ex.quantified_roi_opportunity
        st.metric("Quantified Opportunity", f"${roi:,.2f}" if roi > 0 else "Baseline Stable", delta="Annualized Value")
    with col4:
        st.metric("Domain Profile", report.business_type.value.upper(), delta="Autonomous Detection")

    st.markdown("---")

    # Overview Banner
    st.subheader("Executive Strategic Assessment")
    st.info(f"**AI Strategy Takeaway:** {ex.business_health_overview}")

    # Growth vs Risks Columns
    c_grow, c_risk = st.columns(2)
    with c_grow:
        st.markdown("### 🚀 Primary Growth Catalysts")
        for g in ex.key_growth_drivers:
            st.markdown(f"- **{g}**")

    with c_risk:
        st.markdown("### ⚠️ Key Operational & Retention Risks")
        for r in ex.top_operational_risks:
            st.markdown(f"- **{r}**")

    st.markdown("---")

    # Tactical Playbooks
    st.subheader("🎯 Prioritized Action Playbook")

    p7, p30, p90 = st.tabs(["🚨 7-Day Quick Wins", "📅 30-Day Operational Levers", "🎯 90-Day Strategic Initiatives"])
    with p7:
        if ex.playbook_7_days:
            for item in ex.playbook_7_days:
                with st.expander(f"**[{item.department}]** {item.title} — Impact: ${item.estimated_impact_dollars:,.2f}" if item.estimated_impact_dollars else f"**[{item.department}]** {item.title}", expanded=True):
                    st.write(item.description)
                    st.caption(f"**Expected Outcome:** {item.expected_outcome} | **Difficulty:** {item.difficulty}")
        else:
            st.write("No immediate emergency triage required.")

    with p30:
        if ex.playbook_30_days:
            for item in ex.playbook_30_days:
                with st.expander(f"**[{item.department}]** {item.title} — Impact: ${item.estimated_impact_dollars:,.2f}" if item.estimated_impact_dollars else f"**[{item.department}]** {item.title}", expanded=True):
                    st.write(item.description)
                    st.caption(f"**Expected Outcome:** {item.expected_outcome} | **Difficulty:** {item.difficulty}")
        else:
            st.write("No 30-day initiatives scheduled.")

    with p90:
        if ex.playbook_90_days:
            for item in ex.playbook_90_days:
                with st.expander(f"**[{item.department}]** {item.title} — Impact: ${item.estimated_impact_dollars:,.2f}" if item.estimated_impact_dollars else f"**[{item.department}]** {item.title}", expanded=True):
                    st.write(item.description)
                    st.caption(f"**Expected Outcome:** {item.expected_outcome} | **Difficulty:** {item.difficulty}")
        else:
            st.write("No 90-day initiatives scheduled.")

    st.markdown("---")
    # Export Report
    md_content = generate_markdown_report(report)
    st.download_button(
        label="📄 Download Complete Executive Brief (Markdown)",
        data=md_content,
        file_name=f"OmniInsight_Report_{report.audit_id}.md",
        mime="text/markdown",
        use_container_width=True,
    )

# ----------------- TAB 2: DEMAND FORECAST -----------------
with tab_fc:
    if ModelType.TIME_SERIES.value in models and models[ModelType.TIME_SERIES.value].status == "success":
        fc = models[ModelType.TIME_SERIES.value]
        st.subheader(f"📈 {fc.model_name}")

        for ins in fc.insights:
            st.success(f"💡 {ins}")

        col_fc1, col_fc2 = st.columns([3, 1])
        with col_fc1:
            fig_fc = plot_time_series_forecast(fc.charts_data, target_name=p.detected_target_col or "Revenue")
            st.plotly_chart(fig_fc, use_container_width=True)

        with col_fc2:
            st.markdown("#### Forecast Metrics")
            st.metric("Projected Total", f"${fc.quantitative_findings.get('projected_total', 0):,.2f}")
            st.metric("Expected Growth", f"{fc.quantitative_findings.get('growth_rate_pct', 0)}%")
            st.metric("Model MAPE", f"{fc.metrics.get('MAPE_pct')}%")
            st.metric("Model R² Fit", f"{fc.metrics.get('R2_score')}")

        if "day_of_week_seasonality" in fc.tables:
            st.markdown("---")
            fig_dow = plot_seasonality_dow(fc.tables["day_of_week_seasonality"])
            st.plotly_chart(fig_dow, use_container_width=True)
    else:
        st.warning("Time series forecasting was not applicable or lacked sufficient history for this dataset.")

# ----------------- TAB 3: CUSTOMER RFM -----------------
with tab_rfm:
    if ModelType.CUSTOMER_SEGMENTATION.value in models and models[ModelType.CUSTOMER_SEGMENTATION.value].status == "success":
        seg = models[ModelType.CUSTOMER_SEGMENTATION.value]
        st.subheader(f"👥 {seg.model_name}")

        for ins in seg.insights:
            st.info(f"💡 {ins}")

        st.plotly_chart(plot_customer_segments(seg.charts_data), use_container_width=True)

        if "segment_summary" in seg.tables:
            st.markdown("#### Customer Persona Breakdown")
            df_seg = pd.DataFrame(seg.tables["segment_summary"])
            st.dataframe(df_seg, use_container_width=True)
    else:
        st.warning("Customer segmentation requires a customer identifier attribute.")

# ----------------- TAB 4: CHURN RADAR -----------------
with tab_churn:
    if ModelType.CHURN_PREDICTION.value in models and models[ModelType.CHURN_PREDICTION.value].status == "success":
        ch = models[ModelType.CHURN_PREDICTION.value]
        st.subheader(f"⚠️ {ch.model_name}")

        for ins in ch.insights:
            st.warning(f"🚨 {ins}")

        col_ch1, col_ch2 = st.columns([2, 1])
        with col_ch1:
            st.plotly_chart(plot_churn_drivers(ch.charts_data), use_container_width=True)
        with col_ch2:
            st.markdown("#### Churn Model Accuracy")
            st.metric("ROC-AUC Score", ch.metrics.get("roc_auc"))
            st.metric("Model Recall", f"{float(ch.metrics.get('recall', 0)) * 100:.1f}%")
            st.metric("High-Risk Accounts", ch.quantitative_findings.get("high_risk_accounts_count", 0))

        if "high_risk_watchlist" in ch.tables and ch.tables["high_risk_watchlist"]:
            st.markdown("#### 🚨 High-Risk Customer Watchlist")
            st.dataframe(pd.DataFrame(ch.tables["high_risk_watchlist"]), use_container_width=True)
    else:
        st.info("Churn prediction radar was skipped or not required for this dataset.")

# ----------------- TAB 5: PRICING POWER -----------------
with tab_price:
    if ModelType.PRICE_ELASTICITY.value in models and models[ModelType.PRICE_ELASTICITY.value].status == "success":
        el = models[ModelType.PRICE_ELASTICITY.value]
        st.subheader(f"🏷️ {el.model_name}")

        for ins in el.insights:
            st.success(f"💡 {ins}")

        col_p1, col_p2 = st.columns([2, 1])
        with col_p1:
            st.plotly_chart(plot_price_elasticity_scatter(el.charts_data), use_container_width=True)
        with col_p2:
            st.markdown("#### Portfolio Elasticity")
            st.metric("Elasticity (Ed)", el.metrics.get("overall_elasticity"))
            st.metric("Pricing Status", el.metrics.get("demand_elasticity_type", "").split("(")[0])

        if "category_elasticities" in el.tables and el.tables["category_elasticities"]:
            st.markdown("#### Category-by-Category Pricing Strategy")
            st.dataframe(pd.DataFrame(el.tables["category_elasticities"]), use_container_width=True)
    else:
        st.info("Pricing elasticity requires unit price and quantity attributes.")

# ----------------- TAB 6: ANOMALY RADAR -----------------
with tab_anom:
    if ModelType.ANOMALY_DETECTION.value in models and models[ModelType.ANOMALY_DETECTION.value].status == "success":
        an = models[ModelType.ANOMALY_DETECTION.value]
        st.subheader(f"🛡️ {an.model_name}")

        for ins in an.insights:
            st.error(f"⚠️ {ins}")

        if "anomaly_alerts" in an.tables and an.tables["anomaly_alerts"]:
            st.markdown("#### Top Detected Operational Outliers")
            st.dataframe(pd.DataFrame(an.tables["anomaly_alerts"]), use_container_width=True)
    else:
        st.info("Anomaly detection requires continuous numerical features.")
