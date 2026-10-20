"""
Synthetic business dataset generator for Retail, SaaS, and Restaurant SMBs.
Generates mathematically realistic distributions, seasonality, churn signals,
and pricing elasticities for robust agent analysis and model validation.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

from data_intelligence.config import settings


def generate_retail_ecommerce_data(
    n_transactions: int = 3500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic multi-channel retail / e-commerce sales dataset.
    Features:
    - Customer repeat behaviors (RFM dynamics)
    - Product category price variations and margin structures
    - Holiday seasonal uplift and weekend spikes
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 10, 31)
    date_range_days = (end_date - start_date).days

    # Customer pool (500 distinct customers with power-law order frequency)
    n_customers = 500
    customer_ids = [f"CUST-{1000 + i}" for i in range(n_customers)]
    customer_weights = np.random.pareto(a=1.8, size=n_customers)
    customer_weights /= customer_weights.sum()

    categories = {
        "Electronics": {"price_range": (80.0, 650.0), "margin": 0.28, "elasticity": -1.8},
        "Apparel": {"price_range": (25.0, 160.0), "margin": 0.55, "elasticity": -1.2},
        "Home & Kitchen": {"price_range": (35.0, 280.0), "margin": 0.45, "elasticity": -0.9},
        "Beauty & Wellness": {"price_range": (18.0, 95.0), "margin": 0.62, "elasticity": -0.6},

    pass
