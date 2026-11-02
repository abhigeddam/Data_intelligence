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


@router.post("/audit/file", response_model=AuditReport)
async def audit_uploaded_file(
    file: UploadFile = File(...),
    business_type: Optional[BusinessType] = Form(None),
) -> AuditReport:
    """Upload CSV or Excel file and trigger autonomous multi-agent audit."""
    content = await file.read()
    filename = file.filename or "uploaded_data.csv"

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported format. Please upload a .csv or .xlsx file.",
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

    report = orchestrator.run_from_dataframe(
        df=df,
        dataset_name=filename.rsplit(".", 1)[0],
        business_type=business_type,
    )
    return report


@router.post("/audit/sample/{sample_id}", response_model=AuditReport)
def audit_sample_dataset(sample_id: str) -> AuditReport:
    """Run full multi-agent audit on a preloaded SMB benchmark dataset."""
    sample_key = sample_id.lower()
    if sample_key not in SAMPLE_MAP:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found. Available: {list(SAMPLE_MAP.keys())}")

    meta = SAMPLE_MAP[sample_key]
    file_path = settings.samples_dir / meta["file"]

    if not file_path.exists():
        from data_intelligence.data.generator import save_sample_datasets
        save_sample_datasets()

    return orchestrator.run_from_file(
        file_path=file_path,
        business_type=BusinessType(meta["type"]),
    )


@router.post("/query", response_model=QueryResponse)
def query_data(req: QueryRequest) -> QueryResponse:
    """Answer natural language business question on data."""
    sample_key = (req.sample_dataset or "retail").lower()
    if sample_key not in SAMPLE_MAP:
        sample_key = "retail"

    meta = SAMPLE_MAP[sample_key]
    file_path = settings.samples_dir / meta["file"]
    if not file_path.exists():
        from data_intelligence.data.generator import save_sample_datasets
        save_sample_datasets()

    df = pd.read_csv(file_path)
    profiler = DataProfiler(business_type=BusinessType(meta["type"]))
    profile = profiler.profile(df)

    res = query_agent.answer(question=req.question, df=df, profile=profile)
    return QueryResponse(**res)
