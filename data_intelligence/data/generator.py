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
        total_cost = round(unit_cost * qty, 2)
        net_profit = round(gross_revenue - total_cost, 2)

        records.append({
            "order_id": f"ORD-{20250000 + i}",
            "order_date": tx_date.strftime("%Y-%m-%d"),
            "customer_id": cust_id,
            "product_id": f"PROD-{cat[:3].upper()}-{np.random.randint(100, 150)}",
            "product_category": cat,
            "unit_price": actual_unit_price,
            "quantity": qty,
            "discount_rate": discount_rate,
            "unit_cost": unit_cost,
            "gross_revenue": gross_revenue,
            "net_profit": net_profit,
            "channel": np.random.choice(channels, p=channel_probs),
            "payment_method": np.random.choice(payment_methods),
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="order_date").reset_index(drop=True)
    return df


def generate_saas_subscription_data(
    n_accounts: int = 1200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic B2B SaaS subscription and account retention dataset.
    Features:
    - Tiered pricing (Starter, Growth, Scale, Enterprise)
    - Churn probability tied directly to usage, seat utilization, support tickets, and NPS
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    start_date = datetime(2024, 6, 1)
    end_date = datetime(2026, 10, 31)
    total_days = (end_date - start_date).days

    tiers = {
        "Starter": {"base_mrr": 49.0, "seat_cap": 3, "cac": 280.0},
        "Growth": {"base_mrr": 199.0, "seat_cap": 12, "cac": 750.0},
        "Scale": {"base_mrr": 499.0, "seat_cap": 30, "cac": 1800.0},
        "Enterprise": {"base_mrr": 1499.0, "seat_cap": 100, "cac": 4500.0},
    }
    tier_names = list(tiers.keys())
    tier_probs = [0.45, 0.35, 0.15, 0.05]

    records = []
    for i in range(n_accounts):
        signup_offset = np.random.randint(0, total_days - 60)
        signup_date = start_date + timedelta(days=signup_offset)
        tenure_months = max(1, int((end_date - signup_date).days / 30.4))

        tier = np.random.choice(tier_names, p=tier_probs)
        tier_cfg = tiers[tier]

        seats_purchased = max(1, int(tier_cfg["seat_cap"] * np.random.uniform(0.6, 1.0)))
        seat_utilization = np.random.beta(a=3.0, b=1.5)
        seats_active = max(1, int(seats_purchased * seat_utilization))

        login_frequency_weekly = float(np.random.gamma(shape=3.5, scale=1.2))
        support_tickets_30d = int(np.random.poisson(lam=1.6))
        nps_score = int(np.clip(np.random.normal(loc=7.8, scale=2.1), 0, 10))

        # Churn probability model based on behavioral risk factors
        churn_logit = (
            -2.2
            - 0.35 * (seats_active / seats_purchased)
            - 0.25 * login_frequency_weekly
            + 0.45 * support_tickets_30d
            - 0.30 * (nps_score - 5)
            - 0.04 * tenure_months
        )
        churn_prob = 1.0 / (1.0 + np.exp(-churn_logit))
        churned = bool(np.random.rand() < churn_prob)

        mrr = round(tier_cfg["base_mrr"] * (1.0 + 0.08 * (seats_purchased - 1)), 2)
        cac = round(tier_cfg["cac"] * np.random.uniform(0.85, 1.25), 2)
        lifetime_value = round(mrr * (tenure_months if churned else (tenure_months + 12)), 2)

        records.append({
            "account_id": f"ACCT-{5000 + i}",
            "account_name": f"Company {chr(65 + (i % 26))}{i:03d}",
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "plan_tier": tier,
            "mrr": mrr,
            "tenure_months": tenure_months,
            "seats_purchased": seats_purchased,
            "seats_active": seats_active,
            "seat_utilization_pct": round(seats_active / seats_purchased * 100, 1),
            "weekly_login_frequency": round(login_frequency_weekly, 2),
            "support_tickets_30d": support_tickets_30d,
            "nps_score": nps_score,
            "customer_acquisition_cost": cac,
            "estimated_ltv": lifetime_value,
            "churned": 1 if churned else 0,
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="signup_date").reset_index(drop=True)
    return df


def generate_restaurant_operations_data(
    n_days: int = 650,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic hospitality / restaurant operational and financial dataset.
    Features:
    - Daily covers, check size, labor and food costs
    - Weekend spikes, weather drag, dine-in vs delivery mix
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    start_date = datetime(2025, 1, 1)
    weather_types = ["Sunny", "Cloudy", "Rainy", "Cold Wave", "Mild"]
    weather_weights = [0.45, 0.25, 0.15, 0.08, 0.07]

    records = []
    base_covers = 140
    for day_idx in range(n_days):
        current_date = start_date + timedelta(days=day_idx)
        dow = current_date.strftime("%A")
        is_weekend = dow in ["Friday", "Saturday", "Sunday"]

        # Weather effect
        weather = np.random.choice(weather_types, p=weather_weights)
        weather_multiplier = 1.0
        if weather == "Rainy":
            weather_multiplier = 0.82
        elif weather == "Cold Wave":
            weather_multiplier = 0.78
        elif weather == "Sunny" and is_weekend:
            weather_multiplier = 1.15

        # Weekend cover volume
        day_mult = 1.65 if dow in ["Friday", "Saturday"] else (1.30 if dow == "Sunday" else 0.90)
        daily_growth = 1.0 + (day_idx / n_days) * 0.18  # steady organic growth
        expected_covers = base_covers * day_mult * weather_multiplier * daily_growth
        covers = max(40, int(np.random.normal(expected_covers, scale=18)))

        # Average check size ($32 - $58)
        avg_check = round(np.random.normal(44.0 + (5.0 if is_weekend else 0.0), scale=4.2), 2)
        dine_in_ratio = 0.75 if not (weather in ["Rainy", "Cold Wave"]) else 0.55
        dine_in_covers = int(covers * dine_in_ratio)
        delivery_covers = covers - dine_in_covers

        dine_in_sales = round(dine_in_covers * avg_check, 2)
        delivery_sales = round(delivery_covers * (avg_check * 0.92), 2)
        total_revenue = round(dine_in_sales + delivery_sales, 2)

        # Operational costs
        food_cost_pct = np.random.uniform(0.28, 0.33)
        food_cost = round(total_revenue * food_cost_pct, 2)

        base_labor = 1100.0 if not is_weekend else 1650.0
        labor_cost = round(base_labor + covers * 3.5 + np.random.normal(0, 50), 2)

        marketing_spend = round(np.random.choice([0.0, 50.0, 120.0, 250.0], p=[0.4, 0.3, 0.2, 0.1]), 2)
        fixed_overhead = 450.0  # rent, utilities allocation

        total_costs = round(food_cost + labor_cost + marketing_spend + fixed_overhead, 2)
        operating_profit = round(total_revenue - total_costs, 2)
        table_turnover_rate = round(np.clip(covers / 65.0, 0.8, 3.8), 2)

        records.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "day_of_week": dow,
            "is_weekend": 1 if is_weekend else 0,
            "weather": weather,
            "covers_total": covers,
            "dine_in_covers": dine_in_covers,
            "delivery_covers": delivery_covers,
            "average_check_usd": avg_check,
            "dine_in_sales": dine_in_sales,
            "delivery_sales": delivery_sales,
            "total_revenue": total_revenue,
            "food_cost": food_cost,
            "labor_cost": labor_cost,
            "marketing_spend": marketing_spend,
            "operating_profit": operating_profit,
            "profit_margin_pct": round((operating_profit / total_revenue) * 100, 2),
            "table_turnover_rate": table_turnover_rate,
        })

    df = pd.DataFrame(records)
    return df


def save_sample_datasets(target_dir: Path | None = None) -> Tuple[Path, Path, Path]:
    """
    Generate and save standard sample datasets into the samples directory.
    """
    out_dir = target_dir or settings.samples_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    retail_path = out_dir / "retail_ecommerce_sales.csv"
    saas_path = out_dir / "saas_customer_retention.csv"
    restaurant_path = out_dir / "restaurant_operations.csv"

    df_retail = generate_retail_ecommerce_data()
    df_retail.to_csv(retail_path, index=False)

    df_saas = generate_saas_subscription_data()
    df_saas.to_csv(saas_path, index=False)

    df_rest = generate_restaurant_operations_data()
    df_rest.to_csv(restaurant_path, index=False)

    return retail_path, saas_path, restaurant_path


if __name__ == "__main__":
    p1, p2, p3 = save_sample_datasets()
    print(f"Generated sample datasets:\n- {p1}\n- {p2}\n- {p3}")
