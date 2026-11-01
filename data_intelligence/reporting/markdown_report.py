"""
Executive Markdown Report Generator for OmniInsight SMB Data Intelligence.
Formats complex multi-agent analysis into an executive brief.
"""

from data_intelligence.types import AuditReport, ModelType


def generate_markdown_report(report: AuditReport) -> str:
    """
    Format AuditReport into a publication-grade executive Markdown briefing.
    """
    p = report.dataset_profile
    q = p.data_quality
    ex = report.executive_summary
    models = report.model_results

    md = []
    md.append(f"# 📊 OmniInsight Executive Intelligence Brief")
    md.append(f"**Audit Reference:** `{report.audit_id}` | **Domain Profile:** `{report.business_type.value.upper()}` | **Date:** {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}\n")
    md.append(f"---\n")

    # 1. Executive Summary
    md.append(f"## 1. Executive Strategy Overview")
    md.append(f"> **Overview:** {ex.business_health_overview}\n")
    if ex.quantified_roi_opportunity > 0:
        md.append(f"### 💰 Quantified Growth & Recovery Opportunity: **${ex.quantified_roi_opportunity:,.2f}**\n")

    col_growth = "\n".join([f"- 🚀 {g}" for g in ex.key_growth_drivers])
    col_risks = "\n".join([f"- ⚠️ {r}" for r in ex.top_operational_risks])

    md.append(f"| Key Growth Catalysts | Primary Operational & Churn Risks |")
    md.append(f"| :--- | :--- |")
    md.append(f"| {col_growth.replace(chr(10), '<br>')} | {col_risks.replace(chr(10), '<br>')} |\n")

    # 2. Data Health & Structural Hygiene
    md.append(f"## 2. Dataset Hygiene & Semantic Profile")
    md.append(f"- **Total Records Analyzed:** `{p.shape[0]:,}` rows × `{p.shape[1]}` columns ({p.memory_usage_mb} MB)")
    md.append(f"- **Data Quality Score:** **{q.overall_score:.1f} / 100** (Completeness: {q.completeness_score}%, Uniqueness: {q.uniqueness_score}%, Consistency: {q.consistency_score}%)")
    md.append(f"- **Identified Anchor Metrics:** Datetime (`{p.detected_datetime_col or 'N/A'}`), Target (`{p.detected_target_col or 'N/A'}`), Customer Key (`{p.detected_customer_id_col or 'N/A'}`)\n")

    if q.warnings:
        md.append(f"**Data Hygiene Notes:**")
        for w in q.warnings:
            md.append(f"- {w}")
        md.append("")

    # 3. Validated Business Hypotheses
    md.append(f"## 3. Autonomous Hypotheses & Validation Matrix")
    md.append(f"| Hypothesis ID | Strategic Focus | Priority | Status | Target Metric |")
    md.append(f"| :--- | :--- | :--- | :--- | :--- |")
    for hyp in report.hypotheses:
        m_status = models.get(hyp.recommended_model)
        status_tag = f"✅ Validated ({m_status.status.upper()})" if m_status else "⏸️ Pending"
        md.append(f"| `{hyp.id}` | **{hyp.title}**<br>_{hyp.business_question}_ | `{hyp.priority.value}` | {status_tag} | `{hyp.target_metric}` |")
    md.append("")

    # 4. Multi-Model Analytical Deep Dive
    md.append(f"## 4. Multi-Model Predictive & Diagnostic Findings\n")

    # 4.1 Forecasting
    if ModelType.TIME_SERIES.value in models:
        fc = models[ModelType.TIME_SERIES.value]
        if fc.status == "success":
            md.append(f"### 📈 A. Revenue & Demand Forecast (Next {fc.metrics.get('horizon_days', 30)} Days)")
            for ins in fc.insights:
                md.append(f"- {ins}")
            md.append(f"\n*Model Metrics: MAPE = {fc.metrics.get('MAPE_pct')}%, RMSE = ${fc.metrics.get('RMSE')}, R² = {fc.metrics.get('R2_score')}*\n")

            if "day_of_week_seasonality" in fc.tables:
                md.append(f"**Day of Week Seasonality Breakdown:**\n")
                md.append(f"| Day of Week | Average Daily Velocity |")
                md.append(f"| :--- | :--- |")
                for d in fc.tables["day_of_week_seasonality"]:
                    md.append(f"| {d['day_of_week']} | ${d['average_value']:,.2f} |")
                md.append("")

    # 4.2 Customer Segmentation
    if ModelType.CUSTOMER_SEGMENTATION.value in models:
        seg = models[ModelType.CUSTOMER_SEGMENTATION.value]
        if seg.status == "success":
            md.append(f"### 👥 B. Customer RFM Behavioral Cohorts")
            for ins in seg.insights:
                md.append(f"- {ins}")
            md.append(f"\n*Clustering: {seg.metrics.get('optimal_clusters')} clusters (Silhouette Score: {seg.metrics.get('silhouette_score')})*\n")

            if "segment_summary" in seg.tables:
                md.append(f"| Cohort Persona | Customer Count | % of Accounts | Avg Spend | Total Revenue | Rev Share % |")
                md.append(f"| :--- | :--- | :--- | :--- | :--- | :--- |")
                for s in seg.tables["segment_summary"]:
                    md.append(f"| **{s['persona']}** | {s['customer_count']:,} | {s['customer_pct']}% | ${s['avg_spend_usd']:,.2f} | ${s['total_revenue_usd']:,.2f} | **{s['revenue_share_pct']}%** |")
                md.append("")

    # 4.3 Churn Prediction
    if ModelType.CHURN_PREDICTION.value in models:
        ch = models[ModelType.CHURN_PREDICTION.value]
        if ch.status == "success":
            md.append(f"### ⚠️ C. Customer Retention & Churn Risk Radar")
            for ins in ch.insights:
                md.append(f"- {ins}")
            md.append(f"\n*Classifier Accuracy: ROC-AUC = {ch.metrics.get('roc_auc')}, Precision = {ch.metrics.get('precision')}, Recall = {ch.metrics.get('recall')}*\n")

            if "top_churn_drivers" in ch.tables:
                md.append(f"**Top Churn Drivers by Behavioral Importance:**")
                for dr in ch.tables["top_churn_drivers"]:
                    md.append(f"- **{dr['feature']}:** {dr['importance_pct']}% relative impact")
                md.append("")

    # 4.4 Price Elasticity
    if ModelType.PRICE_ELASTICITY.value in models:
        el = models[ModelType.PRICE_ELASTICITY.value]
        if el.status == "success":
            md.append(f"### 🏷️ D. Econometric Price Elasticity & Pricing Power")
            for ins in el.insights:
                md.append(f"- {ins}")
            md.append("")

            if "category_elasticities" in el.tables and el.tables["category_elasticities"]:
                md.append(f"| Product Category | Elasticity (Ed) | Classification | Avg Price | Strategy |")
                md.append(f"| :--- | :--- | :--- | :--- | :--- |")
                for c in el.tables["category_elasticities"]:
                    md.append(f"| **{c['category']}** | `{c['elasticity_coefficient']}` | {c['demand_classification']} | ${c['avg_price_usd']:,.2f} | {c['recommendation']} |")
                md.append("")

    # 4.5 Anomaly Radar
    if ModelType.ANOMALY_DETECTION.value in models:
        an = models[ModelType.ANOMALY_DETECTION.value]
        if an.status == "success":
            md.append(f"### 🛡️ E. Operational Anomaly & Margin Leakage Radar")
            for ins in an.insights:
                md.append(f"- {ins}")
            md.append("")

    # 5. Prioritized Action Playbook
    md.append(f"## 5. Prioritized Tactical & Strategic Action Playbook\n")

    if ex.playbook_7_days:
        md.append(f"### 🚨 Phase 1: 7-Day Immediate Wins")
        md.append(f"| Action Initiative | Dept | Difficulty | Projected Impact | Rationale & Next Steps |")
        md.append(f"| :--- | :--- | :--- | :--- | :--- |")
        for item in ex.playbook_7_days:
            imp_str = f"${item.estimated_impact_dollars:,.2f}" if item.estimated_impact_dollars else "N/A"
            md.append(f"| **{item.title}** | `{item.department}` | `{item.difficulty}` | **{imp_str}** | {item.description}<br>_Expected Outcome: {item.expected_outcome}_ |")
        md.append("")

    if ex.playbook_30_days:
        md.append(f"### 📅 Phase 2: 30-Day Operational Levers")
        md.append(f"| Action Initiative | Dept | Difficulty | Projected Impact | Rationale & Next Steps |")
        md.append(f"| :--- | :--- | :--- | :--- | :--- |")
        for item in ex.playbook_30_days:
            imp_str = f"${item.estimated_impact_dollars:,.2f}" if item.estimated_impact_dollars else "N/A"
            md.append(f"| **{item.title}** | `{item.department}` | `{item.difficulty}` | **{imp_str}** | {item.description}<br>_Expected Outcome: {item.expected_outcome}_ |")
        md.append("")

    if ex.playbook_90_days:
        md.append(f"### 🎯 Phase 3: 90-Day Strategic Initiatives")
        md.append(f"| Action Initiative | Dept | Difficulty | Projected Impact | Rationale & Next Steps |")
        md.append(f"| :--- | :--- | :--- | :--- | :--- |")
        for item in ex.playbook_90_days:
            imp_str = f"${item.estimated_impact_dollars:,.2f}" if item.estimated_impact_dollars else "N/A"
            md.append(f"| **{item.title}** | `{item.department}` | `{item.difficulty}` | **{imp_str}** | {item.description}<br>_Expected Outcome: {item.expected_outcome}_ |")
        md.append("")

    md.append(f"## 6. Strategic Leadership Verdict")
    md.append(f"> {ex.concluding_verdict}\n")
    md.append(f"---\n*Generated autonomously by OmniInsight AI Agent Platform.*")

    return "\n".join(md)
