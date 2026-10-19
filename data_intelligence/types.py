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


