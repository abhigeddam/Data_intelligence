"""
Pydantic API schemas for request/response payloads.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from data_intelligence.types import BusinessType, AuditReport


class QueryRequest(BaseModel):
    question: str
    sample_dataset: Optional[str] = None  # "retail", "saas", "restaurant"


class QueryResponse(BaseModel):
    question: str
    answer: str
    metric_value: Optional[float] = None
    table_data: Optional[List[Dict[str, Any]]] = None


class SampleInfo(BaseModel):
    id: str
    name: str
    business_type: str
    description: str
    rows: int
    columns: int


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
