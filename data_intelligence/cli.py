"""
Command Line Interface (CLI) for OmniInsight SMB Data Intelligence.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from data_intelligence.agents.orchestrator import AgentOrchestrator
from data_intelligence.agents.query_agent import ConversationalQueryAgent
from data_intelligence.types import BusinessType
from data_intelligence.reporting.markdown_report import generate_markdown_report
from data_intelligence.reporting.exporter import ReportExporter
from data_intelligence.data.generator import save_sample_datasets
from data_intelligence.config import settings


def main():
    parser = argparse.ArgumentParser(
        prog="omni-intel",
        description="OmniInsight: Autonomous Multi-Model Data Intelligence Platform for SMBs",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run multi-agent audit on a CSV or Excel dataset")
    analyze_parser.add_argument("--file", "-f", required=True, help="Path to input data file (.csv, .xlsx)")
    analyze_parser.add_argument(
        "--type", "-t",
        choices=["retail", "saas", "restaurant", "ecommerce", "services", "general"],
        default=None,
        help="Explicit business domain (optional, auto-inferred if omitted)",
    )
    analyze_parser.add_argument("--output", "-o", default=None, help="Save executive Markdown report to custom path")
    analyze_parser.add_argument("--export", action="store_true", help="Save both MD and JSON reports in reports/ directory")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a natural language question about a dataset")
    query_parser.add_argument("--file", "-f", required=True, help="Path to dataset file")
    query_parser.add_argument("--question", "-q", required=True, help="Business question to answer")

    # Generate samples command
    subparsers.add_parser("generate-samples", help="Generate synthetic SMB benchmark datasets")

    # Serve API command
    api_parser = subparsers.add_parser("serve-api", help="Launch FastAPI REST backend")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    api_parser.add_argument("--port", type=int, default=8000, help="Port number")

    # Serve UI command
    ui_parser = subparsers.add_parser("serve-ui", help="Launch Streamlit Web UI")
    ui_parser.add_argument("--port", type=int, default=8501, help="Port number")

    args = parser.parse_args()

    pass
