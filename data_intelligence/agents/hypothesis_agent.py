"""
Business Strategy and Hypothesis Generation Agent.
Formulates actionable empirical questions tailored to the SMB's business model.
"""

from typing import Dict, Any, List
import uuid

from data_intelligence.agents.base import BaseAgent
from data_intelligence.types import (
    BusinessType,
    DatasetProfile,
    Hypothesis,
    HypothesisPriority,
    ModelType,
)


class BusinessStrategyHypothesisAgent(BaseAgent):
    """
    Analyzes the dataset profile and generates domain-specific
    hypotheses for empirical validation.
    """

    def __init__(self):
        super().__init__(
            name="Business Strategy & Hypothesis Agent",
            role="Generate empirical hypotheses aligned with SMB growth, retention, and margins",
        )

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        profile: DatasetProfile = context.get("dataset_profile")
        if not profile:
            raise ValueError("DatasetProfile missing in context.")

        hypotheses: List[Hypothesis] = []
        b_type = profile.business_type

        # 1. Forecasting Hypothesis
        if profile.detected_datetime_col and profile.detected_target_col:
            hypotheses.append(
                Hypothesis(
                    id=f"HYP-{uuid.uuid4().hex[:6].upper()}",
                    title="Revenue & Demand Velocity Forecast",
                    priority=HypothesisPriority.HIGH,
                    business_question=f"Will {profile.detected_target_col} expand or contract over the next 30 to 90 days, and what are the primary day-of-week demand patterns?",
                    rationale="Accurate demand forecasting enables proactive cash flow planning, inventory stocking, and staffing optimization.",
                    target_metric=profile.detected_target_col,
                    recommended_model=ModelType.TIME_SERIES.value,
                )
            )

        # 2. Customer Segmentation Hypothesis
        if profile.detected_customer_id_col:
            hypotheses.append(
                Hypothesis(
                    id=f"HYP-{uuid.uuid4().hex[:6].upper()}",
                    title="Customer Revenue Concentration & VIP Cohorts",
                    priority=HypothesisPriority.CRITICAL,
                    business_question="Is the customer base bifurcated into high-yield VIP champions vs low-frequency one-off buyers (Pareto skew)?",
                    rationale="Pinpointing top-spending cohorts allows personalized retention loyalty campaigns and prevents revenue loss from account decay.",
                    target_metric="Customer Lifetime Spend (Monetary)",
                    recommended_model=ModelType.CUSTOMER_SEGMENTATION.value,
                )
            )

        # 3. Churn Risk Hypothesis
        if any(c in profile.columns for c in ["churned", "churn", "is_churn", "status"]) or profile.detected_customer_id_col:
            hypotheses.append(
                Hypothesis(
                    id=f"HYP-{uuid.uuid4().hex[:6].upper()}",
                    title="Account Retention Vulnerability & Churn Drivers",
                    priority=HypothesisPriority.CRITICAL,
                    business_question="Which accounts exhibit immediate churn risk patterns, and what operational friction points trigger cancellation?",
                    rationale="Customer acquisition costs 5x to 7x more than retention. Mitigating churn protects recurring margin baseline.",
                    target_metric="Churn Probability Rate",
                    recommended_model=ModelType.CHURN_PREDICTION.value,
                )
            )

        # 4. Pricing Power & Elasticity Hypothesis
        if profile.detected_quantity_col or any(p in profile.columns for p in ["unit_price", "price", "avg_check"]):
            hypotheses.append(
                Hypothesis(
                    id=f"HYP-{uuid.uuid4().hex[:6].upper()}",
                    title="Price Sensitivity & Untapped Margin Expansion",
                    priority=HypothesisPriority.HIGH,
                    business_question="Do certain product categories or services have inelastic demand that allows for a 5-8% price optimization without sacrificing volume?",
                    rationale="Pricing is the highest-leverage profit driver; discovering price-insensitive segments directly expands bottom-line net profit.",
                    target_metric="Price Elasticity of Demand (Ed)",
                    recommended_model=ModelType.PRICE_ELASTICITY.value,
                )
            )

        # 5. Operational Anomaly Hypothesis
        hypotheses.append(
            Hypothesis(
                id=f"HYP-{uuid.uuid4().hex[:6].upper()}",
                title="Operational Leakage & Margin Compression Outliers",
                priority=HypothesisPriority.MEDIUM,
                business_question="Are there isolated operational days or transactions where costs spiked, margins compressed, or irregular behavior occurred?",
                rationale="Unsupervised anomaly radar detects inventory shrinkage, billing errors, labor over-scheduling, and unexpected losses.",
                target_metric="Anomaly Outlier Score",
                recommended_model=ModelType.ANOMALY_DETECTION.value,
            )
        )

        context["hypotheses"] = hypotheses
        return context
