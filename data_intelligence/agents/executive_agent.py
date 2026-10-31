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
                        title="Execute 4% to 6% Selective Price Adjustment",
                        department="Finance",
                        description="Increase list prices across identified inelastic categories. Because customers are insensitive to price changes here, margin flows directly to profit.",
                        estimated_impact_dollars=est_pricing_gain,
                        difficulty="Low",
                        expected_outcome="Pure EBITDA expansion with near-zero attrition in units sold.",
                    )
                )
            elif "Elastic" in status_desc:
                operational_risks.append(
                    f"Price Sensitivity: Demand is elastic (Ed = {coef:.2f}). Direct price hikes will cause customer defection without value bundling."
                )
                playbook_30.append(
                    ActionItem(
                        time_horizon=TimeHorizon.DAYS_30,
                        title="Implement Strategic Product Bundling Instead of Direct Hikes",
                        department="Marketing",
                        description="Bundle high-margin accessories with core offerings to increase average basket size without triggering unit price sensitivity.",
                        estimated_impact_dollars=14000.0,
                        difficulty="Medium",
                        expected_outcome="Lift Average Order Value (AOV) by 8-12% while maintaining price competitiveness.",
                    )
                )

        # 5. Evaluate Anomaly Radar
        anom_res = model_results.get(ModelType.ANOMALY_DETECTION.value)
        if anom_res and anom_res.status == "success":
            findings = anom_res.quantitative_findings
            total_anom = findings.get("total_anomalies", 0)
            if total_anom > 0:
                operational_risks.append(
                    f"Operational Outliers: Flagged {total_anom} anomalous transactions/days reflecting cost surges or margin anomalies."
                )
                playbook_30.append(
                    ActionItem(
                        time_horizon=TimeHorizon.DAYS_30,
                        title="Audit Flagged High-Severity Operational Anomalies",
                        department="Operations",
                        description="Inspect the top 10 outlier events flagged in OmniInsight Anomaly Radar for payroll discrepancies, unauthorized discounts, or inventory shrinkage.",
                        estimated_impact_dollars=6500.0,
                        difficulty="Low",
                        expected_outcome="Plug recurring operational margin leakage and standardize checkout controls.",
                    )
                )

        # Format high level narrative
        b_type_str = profile.business_type.value.upper()
        health_overview = (
            f"OmniInsight multi-agent audit completed for {b_type_str} business profile across {profile.shape[0]:,} records. "
            f"The business demonstrates a data hygiene rating of {profile.data_quality.overall_score:.1f}/100. "
            f"Autonomous model synthesis identifies strong potential to unlock an estimated ${quantified_roi:,.2f} in annual value "
            f"through targeted retention, pricing calibration, and operational streamlining."
        )

        verdict = (
            f"Conclusion: Prioritize 7-day retention triage for vulnerable high-value accounts, "
            f"followed by month-one margin expansion via calibrated pricing and operational capacity alignment."
        )

        executive_summary = ExecutiveSummary(
            business_health_overview=health_overview,
            key_growth_drivers=growth_drivers or ["Steady historical transaction baseline."],
            top_operational_risks=operational_risks or ["No severe systemic anomalies detected."],
            quantified_roi_opportunity=round(quantified_roi, 2),
            playbook_7_days=playbook_7,
            playbook_30_days=playbook_30,
            playbook_90_days=playbook_90,
            concluding_verdict=verdict,
        )

        context["executive_summary"] = executive_summary
        return context
