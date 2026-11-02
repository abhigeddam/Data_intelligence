"""
API Route Handlers for OmniInsight.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
import pandas as pd
import io

from data_intelligence.api.schemas import (
    QueryRequest,
    QueryResponse,
    SampleInfo,
    HealthResponse,
)
from data_intelligence.types import BusinessType, AuditReport
from data_intelligence.agents.orchestrator import AgentOrchestrator
from data_intelligence.agents.query_agent import ConversationalQueryAgent
from data_intelligence.core.profiler import DataProfiler
from data_intelligence.config import settings

router = APIRouter(prefix="/api/v1")
orchestrator = AgentOrchestrator()
query_agent = ConversationalQueryAgent()

SAMPLE_MAP = {
    "retail": {
        "file": "retail_ecommerce_sales.csv",
        "name": "Retail & E-Commerce Transactions",
        "type": "retail",
        "desc": "3,500 multi-channel retail orders with products, categories, margins, discounts, and customer IDs.",
    },
    "saas": {
        "file": "saas_customer_retention.csv",
        "name": "SaaS Customer Subscriptions & Retention",
        "type": "saas",
        "desc": "1,200 B2B subscription accounts with MRR, seat usage, login frequency, support tickets, and churn labels.",
    },
    "restaurant": {
        "file": "restaurant_operations.csv",
        "name": "Hospitality & Restaurant Operations",
        "type": "restaurant",
        "desc": "650 operational days with covers, average check, dine-in vs delivery sales, labor, food costs, and weather.",
    },
}


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        app=settings.app_name,
        version=settings.app_version,
    )


@router.get("/samples", response_model=List[SampleInfo])
def list_samples() -> List[SampleInfo]:
    """List preloaded SMB benchmark datasets."""
    samples = []
    for key, meta in SAMPLE_MAP.items():
        file_path = settings.samples_dir / meta["file"]
        if file_path.exists():
            df = pd.read_csv(file_path)
            samples.append(
                SampleInfo(
                    id=key,
                    name=meta["name"],
                    business_type=meta["type"],
                    description=meta["desc"],
                    rows=len(df),
                    columns=len(df.columns),
                )
            )
    return samples
