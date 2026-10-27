"""
Price Elasticity of Demand and Revenue Optimization Simulator.
Estimates econometric price sensitivities across product categories,
identifies pricing power, and calculates revenue-maximizing price adjustments.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression

from data_intelligence.types import ModelType, ModelResult, DatasetProfile


class PriceElasticityEngine:
    """
    Automated econometric price elasticity model for SMB pricing strategy.
    """

    PRICE_KEYWORDS = ["price", "unit_price", "rate", "check", "avg_check"]
    QTY_KEYWORDS = ["qty", "quantity", "units", "volume", "covers", "orders"]

    def __init__(self, profile: DatasetProfile):
        self.profile = profile

    def run(
        self,
        df: pd.DataFrame,
        price_col: Optional[str] = None,
        quantity_col: Optional[str] = None,
        category_col: Optional[str] = None,
    ) -> ModelResult:
        """
        Execute log-log price elasticity regression across categories or overall portfolio.
        """
        start_time = time.time()
        p_col = price_col or self._detect_price_column(df)
        q_col = quantity_col or self._detect_quantity_column(df)
        cat_col = category_col or self.profile.detected_category_col

        if not p_col or not q_col or p_col not in df.columns or q_col not in df.columns:
            return ModelResult(
                model_type=ModelType.PRICE_ELASTICITY,
                model_name="Econometric Price Elasticity Simulator",
                status="skipped",
                insights=["Price elasticity skipped: Requires both unit price and quantity/volume columns."],
                error_message="Price or quantity column not found.",
                execution_time_seconds=round(time.time() - start_time, 2),
            )

        try:
            work_df = df[[p_col, q_col] + ([cat_col] if cat_col and cat_col in df.columns else [])].dropna().copy()
            work_df = work_df[(work_df[p_col] > 0) & (work_df[q_col] > 0)]

            if len(work_df) < 20:
                return ModelResult(
                    model_type=ModelType.PRICE_ELASTICITY,
                    model_name="Econometric Price Elasticity Simulator",
                    status="warning",
                    insights=["Insufficient transaction volume (< 20 rows) to estimate price elasticity."],
                    execution_time_seconds=round(time.time() - start_time, 2),
                )

            # Analyze overall portfolio elasticity
            overall_elasticity, r2_overall = self._calculate_elasticity(work_df[p_col], work_df[q_col])

            # Analyze by category if category column exists
            category_results = []
            if cat_col and cat_col in work_df.columns:
                top_categories = work_df[cat_col].value_counts().head(8).index.tolist()
                for cat in top_categories:
                    sub_df = work_df[work_df[cat_col] == cat]
                    if len(sub_df) >= 15 and sub_df[p_col].nunique() > 2:
                        e_val, r2_val = self._calculate_elasticity(sub_df[p_col], sub_df[q_col])
                        avg_p = float(sub_df[p_col].mean())
                        total_q = float(sub_df[q_col].sum())

                        # Pricing diagnosis
                        if e_val > -0.8:
                            diagnosis = "Inelastic (Strong Pricing Power)"
                            rec = "Can safely increase prices by 5% to 8% to boost margins."
                        elif e_val < -1.4:
                            diagnosis = "Highly Elastic (Price Sensitive)"
                            rec = "Price increases will hurt revenue; optimize bundling or selective promos."
                        else:
                            diagnosis = "Unitary Elastic"
                            rec = "Pricing is balanced with current market demand."

                        category_results.append({
                            "category": str(cat),
                            "elasticity_coefficient": round(e_val, 3),
                            "demand_classification": diagnosis,
                            "r2_fit": round(r2_val, 2),
                            "avg_price_usd": round(avg_p, 2),
                            "total_units_sold": int(total_q),
                            "recommendation": rec,
                        })

            # Formulate strategic narrative
            if overall_elasticity > -0.9:
                portfolio_status = "Inelastic (Pricing Power Available)"
                portfolio_rec = "Your customer base exhibits low price sensitivity. A modest 3%–5% price increase across core offerings will flow directly to net profit with negligible drop in transaction volume."
            elif overall_elasticity < -1.3:
                portfolio_status = "Elastic (High Price Sensitivity)"
                portfolio_rec = "Demand is sensitive to price hikes. Prioritize margin expansion through cross-selling, premium tiering, and supplier cost reduction rather than across-the-board price raises."
            else:
                portfolio_status = "Unitary Elastic (Fair Market Balance)"
                portfolio_rec = "Demand shifts proportionately with price adjustments. Target price adjustments dynamically by customer segment."

            insights = [
                f"Overall Portfolio Elasticity: {overall_elasticity:.2f} — classified as '{portfolio_status}'.",
                portfolio_rec,
            ]

            if category_results:
                inelastic_cats = [c["category"] for c in category_results if "Inelastic" in c["demand_classification"]]
                elastic_cats = [c["category"] for c in category_results if "Highly Elastic" in c["demand_classification"]]

                if inelastic_cats:
                    insights.append(f"Immediate Margin Opportunity: {', '.join(inelastic_cats)} have high pricing resilience and can absorb price increases.")
                if elastic_cats:
                    insights.append(f"Caution Zone: {', '.join(elastic_cats)} show high elasticity; avoid price increases without value bundling.")

            return ModelResult(
                model_type=ModelType.PRICE_ELASTICITY,
                model_name="Econometric Price Elasticity Simulator",
                status="success",
                metrics={
                    "overall_elasticity": round(overall_elasticity, 3),
                    "model_r2_fit": round(r2_overall, 3),
                    "demand_elasticity_type": portfolio_status,
                },
                insights=insights,
                quantitative_findings={
                    "elasticity_coefficient": round(overall_elasticity, 3),
                    "classification": portfolio_status,
                    "analyzed_categories_count": len(category_results),
                },
                tables={
                    "category_elasticities": category_results,
                },
                charts_data={
                    "category_names": [c["category"] for c in category_results],
                    "elasticities": [c["elasticity_coefficient"] for c in category_results],
                    "scatter_p_vs_q": [
                        {"price": float(p), "quantity": float(q)}
                        for p, q in zip(work_df[p_col].sample(min(200, len(work_df)), random_state=42),
                                       work_df[q_col].sample(min(200, len(work_df)), random_state=42))
                    ],
                },
                execution_time_seconds=round(time.time() - start_time, 2),
            )
        except Exception as e:
            return ModelResult(
                model_type=ModelType.PRICE_ELASTICITY,
                model_name="Econometric Price Elasticity Simulator",
                status="error",
                error_message=str(e),
                insights=[f"Price elasticity analysis failed: {str(e)}"],
                execution_time_seconds=round(time.time() - start_time, 2),
            )

    def _calculate_elasticity(self, price: pd.Series, quantity: pd.Series) -> Tuple[float, float]:
        """
        Fit log-log ordinary least squares regression: ln(Q) = a + e * ln(P)
        """
        log_p = np.log(price.values).reshape(-1, 1)
        log_q = np.log(quantity.values)

        reg = LinearRegression()
        reg.fit(log_p, log_q)
        elasticity = float(reg.coef_[0])
        r2 = float(reg.score(log_p, log_q))
        return elasticity, r2

    def _detect_price_column(self, df: pd.DataFrame) -> Optional[str]:
        for c in df.columns:
            if any(k == c.lower() or k in c.lower() for k in ["unit_price", "price", "avg_check", "rate", "cost_per_unit"]):
                return c
        return None

    def _detect_quantity_column(self, df: pd.DataFrame) -> Optional[str]:
        for c in df.columns:
            if any(k == c.lower() or k in c.lower() for k in ["quantity", "qty", "covers_total", "volume", "units", "covers"]):
                return c
        return None
