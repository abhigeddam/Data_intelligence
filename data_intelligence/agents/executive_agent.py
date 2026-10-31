"""
Executive Narrative and Action Agent (AI Chief Strategy Officer).
Synthesizes mathematical models and statistical findings into plain-English
business strategy, quantified financial opportunities, and a prioritized action playbook.
"""

from typing import Dict, Any, List
import pandas as pd

from data_intelligence.agents.base import BaseAgent
from data_intelligence.types import (
    ExecutiveSummary,
    ActionItem,
    TimeHorizon,
    ModelResult,
    ModelType,
    DatasetProfile,
)


class ExecutiveNarrativeAgent(BaseAgent):
    """
    Acts as an autonomous virtual CFO / Chief Strategy Officer.
    Translates model findings into prioritized growth and risk mitigation playbooks.
    """

    def __init__(self):
        super().__init__(
            name="Executive Narrative & Action Agent",
            role="Synthesize analytical findings into quantified executive business strategy and action playbooks",
        )

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        profile: DatasetProfile = context.get("dataset_profile")
        model_results: Dict[str, ModelResult] = context.get("model_results", {})
        cleaned_df: pd.DataFrame = context.get("cleaned_df")

        # Extract findings from models
        growth_drivers: List[str] = []
        operational_risks: List[str] = []
        quantified_roi: float = 0.0

        playbook_7: List[ActionItem] = []
        playbook_30: List[ActionItem] = []
        playbook_90: List[ActionItem] = []

        # 1. Evaluate Time Series Forecasting
        fc_res = model_results.get(ModelType.TIME_SERIES.value)
        if fc_res and fc_res.status == "success":
            findings = fc_res.quantitative_findings
            projected_rev = findings.get("projected_total", 0.0)
            growth_pct = findings.get("growth_rate_pct", 0.0)
            top_day = findings.get("top_day", "Weekend")

            if growth_pct >= 0:
                growth_drivers.append(
                    f"Demand Momentum: Projected 30-day expansion of +{growth_pct}% (${projected_rev:,.2f}), led by high {top_day} footfall/conversions."
                )
                playbook_30.append(
                    ActionItem(
                        time_horizon=TimeHorizon.DAYS_30,
                        title=f"Scale Capacity & Staffing for {top_day} Peak Demand",
                        department="Operations",
                        description=f"Shift employee scheduling and inventory prep toward {top_day}s to capture maximum peak throughput.",
                        estimated_impact_dollars=round(projected_rev * 0.04, 2),

        return None
