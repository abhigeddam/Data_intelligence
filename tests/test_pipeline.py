"""
End-to-End Integration Pipeline Tests.
"""

from data_intelligence.agents.orchestrator import AgentOrchestrator
from data_intelligence.reporting.markdown_report import generate_markdown_report
from data_intelligence.reporting.exporter import ReportExporter
from data_intelligence.config import settings


def test_end_to_end_retail_pipeline():
    orchestrator = AgentOrchestrator()
    sample_file = settings.samples_dir / "retail_ecommerce_sales.csv"
    assert sample_file.exists()

    report = orchestrator.run_from_file(sample_file)

    assert report.audit_id.startswith("AUDIT-")
    assert report.dataset_profile.data_quality.overall_score >= 80.0
    assert len(report.hypotheses) >= 4
    assert len(report.model_results) >= 4
    assert len(report.executive_summary.key_growth_drivers) > 0

    # Test report generation
    md_content = generate_markdown_report(report)
    assert "# 📊 OmniInsight Executive Intelligence Brief" in md_content
    assert "Executive Strategy Overview" in md_content

    # Test export
    exporter = ReportExporter()
    md_p, js_p = exporter.export(report)
    assert md_p.exists()
    assert js_p.exists()


def test_end_to_end_saas_pipeline():
    orchestrator = AgentOrchestrator()
    sample_file = settings.samples_dir / "saas_customer_retention.csv"
    assert sample_file.exists()

    report = orchestrator.run_from_file(sample_file)
    assert report.business_type.value == "saas"
    assert "churn_prediction" in report.model_results
