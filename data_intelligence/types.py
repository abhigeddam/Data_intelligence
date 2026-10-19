"""
Data types, schemas, and domain models for OmniInsight Data Intelligence.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class BusinessType(str, Enum):
    RETAIL = "retail"
    SAAS = "saas"
    RESTAURANT = "restaurant"
    ECOMMERCE = "ecommerce"
    SERVICES = "services"
    GENERAL = "general"


class ColumnSemanticType(str, Enum):
    DATETIME = "datetime"
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"
    TEXT = "text"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class ColumnProfile(BaseModel):
    name: str
    semantic_type: ColumnSemanticType
    dtype: str
    total_count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    is_constant: bool = False
    is_unique: bool = False
    stats: Dict[str, Any] = Field(default_factory=dict)
    sample_values: List[Any] = Field(default_factory=list)


class DataQualityScore(BaseModel):
    overall_score: float = Field(ge=0.0, le=100.0)
    completeness_score: float = Field(ge=0.0, le=100.0)
    uniqueness_score: float = Field(ge=0.0, le=100.0)
    consistency_score: float = Field(ge=0.0, le=100.0)
    total_rows: int
    total_columns: int
    total_duplicates: int
    columns_with_missing: List[str] = Field(default_factory=list)
    critical_issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


