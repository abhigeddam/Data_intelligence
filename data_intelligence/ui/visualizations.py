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
