"""
Export utilities for Audit Reports (Markdown and JSON formats).
"""

from pathlib import Path
from typing import Tuple
import json

from data_intelligence.types import AuditReport
from data_intelligence.reporting.markdown_report import generate_markdown_report
from data_intelligence.config import settings


class ReportExporter:
    """
    Saves Audit Reports in Markdown and JSON formats.
    """

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or settings.reports_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, report: AuditReport) -> Tuple[Path, Path]:
        """
        Export both Markdown and JSON files for an AuditReport.
        """
        stem = f"{report.audit_id}_{report.business_type.value}_{report.dataset_name}"
        md_path = self.output_dir / f"{stem}.md"
        json_path = self.output_dir / f"{stem}.json"

        # Generate markdown content
        md_content = generate_markdown_report(report)
        md_path.write_text(md_content, encoding="utf-8")

        # Generate JSON content
        json_content = report.model_dump_json(indent=2)
        json_path.write_text(json_content, encoding="utf-8")

        return md_path, json_path
