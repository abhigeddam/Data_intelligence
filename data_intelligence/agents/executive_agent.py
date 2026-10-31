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
                        difficulty="Low",
                        expected_outcome="Reduce customer wait times and prevent stockouts during peak volume.",
                    )
                )
            else:
                operational_risks.append(
                    f"Demand Contraction Warning: Trailing velocity points to a {abs(growth_pct)}% drop in 30-day top-line run rate."
                )
                playbook_7.append(
                    ActionItem(
                        time_horizon=TimeHorizon.DAYS_7,
                        title="Launch Demand Re-ignition Campaign",
                        department="Marketing",
                        description="Deploy an immediate limited-time promotion across email and SMS channels to counter projected run-rate slowdown.",
                        estimated_impact_dollars=round(projected_rev * 0.06, 2),
                        difficulty="Low",
                        expected_outcome="Stabilize demand baseline and reverse the forecasted velocity dip.",
                    )
                )

        # 2. Evaluate Customer Segmentation
        seg_res = model_results.get(ModelType.CUSTOMER_SEGMENTATION.value)
        if seg_res and seg_res.status == "success":
            findings = seg_res.quantitative_findings
            top_seg_name = findings.get("top_segment_name", "Champions")
            top_seg_share = findings.get("top_segment_rev_share", 0.0)

            growth_drivers.append(
                f"VIP Revenue Anchor: The '{top_seg_name}' cohort delivers {top_seg_share}% of total business revenue."
            )
            playbook_7.append(
                ActionItem(
                    time_horizon=TimeHorizon.DAYS_7,
                    title=f"Deploy Concierge Outreach to '{top_seg_name}' Accounts",
                    department="Sales",
                    description="Export the VIP customer list from OmniInsight and initiate direct, personalized appreciation touchpoints.",
                    estimated_impact_dollars=12000.0,
                    difficulty="Low",
                    expected_outcome="Lock in customer retention for the highest-yield segment and identify upsell opportunities.",
                )
            )
            playbook_90.append(
                ActionItem(
                    time_horizon=TimeHorizon.DAYS_90,
                    title="Formalize Tiered VIP Loyalty & Incentive Program",
                    department="Marketing",
                    description="Design structured rewards (exclusive perks, early access, volume pricing) to nurture repeat purchase frequency.",
                    estimated_impact_dollars=24000.0,
                    difficulty="Medium",
                    expected_outcome="Increase customer lifetime value (LTV) by an estimated 15% across recurring buyers.",
                )
            )

        # 3. Evaluate Churn Prediction
        churn_res = model_results.get(ModelType.CHURN_PREDICTION.value)
        if churn_res and churn_res.status == "success":
            findings = churn_res.quantitative_findings
            rev_at_risk = findings.get("revenue_at_risk_usd", 0.0)
            high_risk_count = findings.get("high_risk_accounts_count", 0)
            driver = findings.get("primary_churn_driver", "Usage Inactivity")

            if rev_at_risk > 0:
                operational_risks.append(
                    f"Retention Vulnerability: {high_risk_count} accounts show imminent churn risk, threatening ${rev_at_risk:,.2f} in revenue."
                )
                quantified_roi += rev_at_risk * 0.40  # Assume 40% can be saved

                playbook_7.append(
                    ActionItem(
                        time_horizon=TimeHorizon.DAYS_7,
                        title=f"Emergency Interventions for Top {min(15, high_risk_count)} At-Risk Accounts",
                        department="Customer Success",
                        description=f"Direct customer support team to reach out to the watchlist accounts with friction points related to '{driver}'.",
                        estimated_impact_dollars=round(rev_at_risk * 0.40, 2),
                        difficulty="Medium",
                        expected_outcome=f"Save up to 40% of the ${rev_at_risk:,.2f} threatened revenue within 14 days.",
                    )
                )

        # 4. Evaluate Price Elasticity
        elast_res = model_results.get(ModelType.PRICE_ELASTICITY.value)
        if elast_res and elast_res.status == "success":
            findings = elast_res.quantitative_findings
            status_desc = findings.get("classification", "Balanced")
            coef = findings.get("elasticity_coefficient", -1.0)

            if "Inelastic" in status_desc:
                growth_drivers.append(
                    f"Pricing Power Unlocked: Demand is inelastic (Ed = {coef:.2f}), confirming low customer price sensitivity."
                )
                est_pricing_gain = 18500.0
                quantified_roi += est_pricing_gain
                playbook_30.append(
                    ActionItem(
                        time_horizon=TimeHorizon.DAYS_30,

        return None
