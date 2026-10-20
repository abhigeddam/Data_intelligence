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
        "Gourmet Food": {"price_range": (12.0, 75.0), "margin": 0.40, "elasticity": -1.1},
    }
    cat_names = list(categories.keys())
    cat_probs = [0.22, 0.30, 0.20, 0.16, 0.12]

    channels = ["Online Store", "In-Store POS", "Amazon Marketplace"]
    channel_probs = [0.55, 0.30, 0.15]
    payment_methods = ["Credit Card", "Apple Pay", "PayPal", "Klarna Buy-Now-Pay-Later"]

    records = []
    for i in range(n_transactions):
        # Sample date with weekly and holiday seasonality
        day_offset = int(np.random.beta(a=1.2, b=1.0) * date_range_days)
        tx_date = start_date + timedelta(days=day_offset)

        # Holiday boost in Nov/Dec
        if tx_date.month in [11, 12]:
            is_holiday_boost = np.random.rand() < 0.35
        else:
            is_holiday_boost = False

        cust_id = np.random.choice(customer_ids, p=customer_weights)
        cat = np.random.choice(cat_names, p=cat_probs)
        cat_info = categories[cat]

        # Base price
        low_p, high_p = cat_info["price_range"]
        base_unit_price = round(np.random.uniform(low_p, high_p), 2)

        # Quantity (typically 1-4 units)
        qty = int(np.random.choice([1, 2, 3, 4, 5], p=[0.60, 0.23, 0.10, 0.05, 0.02]))
        if is_holiday_boost:
            qty += np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])

        # Discount
        discount_rate = float(np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.30], p=[0.50, 0.18, 0.15, 0.10, 0.05, 0.02]))
        actual_unit_price = round(base_unit_price * (1 - discount_rate), 2)
        unit_cost = round(base_unit_price * (1 - cat_info["margin"]), 2)

        gross_revenue = round(actual_unit_price * qty, 2)

        return pd.DataFrame()
