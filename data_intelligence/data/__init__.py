"""
Data generation and loading utilities for OmniInsight SMB Data Intelligence.
"""

from data_intelligence.data.generator import (
    generate_retail_ecommerce_data,
    generate_saas_subscription_data,
    generate_restaurant_operations_data,
    save_sample_datasets,
)

__all__ = [
    "generate_retail_ecommerce_data",
    "generate_saas_subscription_data",
    "generate_restaurant_operations_data",
    "save_sample_datasets",
]
