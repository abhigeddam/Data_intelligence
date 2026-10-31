"""
Multi-Agent Orchestrator for Autonomous SMB Data Intelligence.
Coordinates sequential execution of Profiler, Hypothesis, Specialist,
and Executive Narrative agents into a unified, high-value AuditReport.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import uuid
import pandas as pd
from datetime import datetime, timezone

from data_intelligence.agents.profiler_agent import DataProfilerAgent
from data_intelligence.agents.hypothesis_agent import BusinessStrategyHypothesisAgent
from data_intelligence.agents.specialist_agent import ModelEnsembleSpecialistAgent
from data_intelligence.agents.executive_agent import ExecutiveNarrativeAgent
from data_intelligence.agents.query_agent import ConversationalQueryAgent
from data_intelligence.types import BusinessType, AuditReport


class AgentOrchestrator:
    """
    Autonomous multi-agent orchestration engine.
    """

    def __init__(self, business_type: Optional[BusinessType] = None):
        self.business_type = business_type
        self.profiler_agent = DataProfilerAgent(business_type=business_type)
        self.hypothesis_agent = BusinessStrategyHypothesisAgent()
        self.specialist_agent = ModelEnsembleSpecialistAgent()
        self.executive_agent = ExecutiveNarrativeAgent()
        self.query_agent = ConversationalQueryAgent()

    def run_from_dataframe(
        self,
        df: pd.DataFrame,
        dataset_name: str = "business_data",
        business_type: Optional[BusinessType] = None,
    ) -> AuditReport:
        """
        Execute full multi-agent audit on a pandas DataFrame.
        """
        audit_id = f"AUDIT-{uuid.uuid4().hex[:8].upper()}"
        b_type = business_type or self.business_type

        context: Dict[str, Any] = {
            "audit_id": audit_id,
            "dataset_name": dataset_name,
            "business_type": b_type,
            "raw_df": df,
        }

        # Step 1: Profiling & Sanitization
        context = self.profiler_agent.run(context)

        # Step 2: Hypothesis Generation
        context = self.hypothesis_agent.run(context)

        # Step 3: Multi-Model Execution
        context = self.specialist_agent.run(context)

        # Step 4: Executive Synthesis & Action Playbook
        context = self.executive_agent.run(context)

        # Instantiate AuditReport
        report = AuditReport(
            audit_id=audit_id,
            dataset_name=dataset_name,
            business_type=context["business_type"],
            generated_at=datetime.now(timezone.utc),
            dataset_profile=context["dataset_profile"],
            hypotheses=context["hypotheses"],
            model_results=context["model_results"],
            executive_summary=context["executive_summary"],
        )

        return report

    def run_from_file(
        self,
        file_path: str | Path,
        business_type: Optional[BusinessType] = None,
    ) -> AuditReport:
        """
        Load a CSV or Excel file and run multi-agent intelligence.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        dataset_name = path.stem
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Expected .csv or .xlsx")

        return self.run_from_dataframe(
            df=df,
            dataset_name=dataset_name,
            business_type=business_type,
        )
