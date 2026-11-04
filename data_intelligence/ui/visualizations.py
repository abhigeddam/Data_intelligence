"""
Plotly visualization builders for OmniInsight Streamlit Dashboard.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def plot_time_series_forecast(charts_data: Dict[str, Any], target_name: str = "Metric") -> go.Figure:
    """
    Generate interactive Plotly time series chart with historical, fitted, and forecast with confidence bands.
    """
    fig = go.Figure()

    # Historical Actuals
    hist_dates = charts_data.get("historical_dates", [])
    hist_actuals = charts_data.get("historical_actuals", [])
    hist_fitted = charts_data.get("historical_fitted", [])

    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_actuals,
        mode="lines",
        name="Historical Actual",
        line=dict(color="#2563EB", width=2),
    ))

    # Fitted line
    if hist_fitted:
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_fitted,
            mode="lines",
            name="Model Baseline (Fitted)",
            line=dict(color="#93C5FD", width=1.5, dash="dot"),
        ))

    # Forecast
    fc_dates = charts_data.get("forecast_dates", [])
    fc_vals = charts_data.get("forecast_values", [])
    lb = charts_data.get("lower_bound", [])
    ub = charts_data.get("upper_bound", [])

    if fc_dates and fc_vals:
        # Confidence interval band
        if lb and ub:
            fig.add_trace(go.Scatter(
                x=fc_dates + fc_dates[::-1],
                y=ub + lb[::-1],
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% Confidence Band",
                hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=fc_dates,
            y=fc_vals,
            mode="lines+markers",
            name="Projected Forecast",
            line=dict(color="#059669", width=2.5, dash="dash"),
            marker=dict(size=4),
        ))

    fig.update_layout(
        title=f"Autonomous Demand Forecast: {target_name}",
        xaxis_title="Timeline",
        yaxis_title=f"{target_name} ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_seasonality_dow(dow_table: List[Dict[str, Any]]) -> go.Figure:
    """
    Day of week velocity bar chart.
    """
    df = pd.DataFrame(dow_table)
    fig = px.bar(
        df,
        x="day_of_week",
        y="average_value",
        color="average_value",
        color_continuous_scale="Blues",
        labels={"day_of_week": "Day of Week", "average_value": "Avg Velocity ($)"},
        title="Day-of-Week Seasonality Pattern",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig


def plot_customer_segments(charts_data: Dict[str, Any]) -> go.Figure:
    """
    Customer RFM Recency vs Monetary scatter plot colored by persona.
    """
    data = charts_data.get("recency_vs_monetary", [])
    if not data:
        return go.Figure()

    df = pd.DataFrame(data)
    fig = px.scatter(
        df,
        x="recency",
        y="monetary",
        color="persona",
        size="frequency",
        hover_data=["frequency"],
        labels={"recency": "Days Since Last Transaction (Recency)", "monetary": "Cumulative Spend ($)"},
        title="Customer Behavioral Segments (RFM Clustering)",
        log_y=True,
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_churn_drivers(charts_data: Dict[str, Any]) -> go.Figure:
    """
    Feature importance bar chart for churn predictors.
    """
    names = charts_data.get("driver_names", [])
    imps = charts_data.get("driver_importances", [])
    if not names:
        return go.Figure()

    df = pd.DataFrame({"Feature": names, "Importance": imps}).sort_values(by="Importance", ascending=True)
    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        labels={"Importance": "Relative Impact (%)", "Feature": "Behavioral Factor"},
        title="Top Churn Leading Indicators (Predictive Importance)",
        color="Importance",
        color_continuous_scale="Reds",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig


def plot_price_elasticity_scatter(charts_data: Dict[str, Any]) -> go.Figure:
    """
    Price vs Quantity demand elasticity scatter plot.
    """
    pts = charts_data.get("scatter_p_vs_q", [])
    if not pts:
        return go.Figure()

    df = pd.DataFrame(pts)
    fig = px.scatter(
        df,
        x="price",
        y="quantity",
        labels={"price": "Unit Price ($)", "quantity": "Units Sold / Demand"},
        title="Price vs Demand Elasticity Curve",
        trendline="ols",
        opacity=0.6,
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    return fig
