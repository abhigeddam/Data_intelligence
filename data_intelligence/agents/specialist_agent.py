"""
Model Ensemble Specialist Agent.
Selects, tunes, executes, and evaluates statistical and machine learning models.
"""

from typing import Dict, Any, List
import pandas as pd

from data_intelligence.agents.base import BaseAgent
from data_intelligence.models.forecaster import TimeSeriesForecaster
from data_intelligence.models.segmentation import CustomerSegmentationEngine
from data_intelligence.models.churn import ChurnPredictionEngine
from data_intelligence.models.elasticity import PriceElasticityEngine
from data_intelligence.models.anomaly import AnomalyDetectionEngine
from data_intelligence.types import DatasetProfile, ModelResult, ModelType


class ModelEnsembleSpecialistAgent(BaseAgent):
    """
    Coordinates multi-model execution, ensures parameter validation,
    and consolidates analytical findings.
    """

    def __init__(self):
        super().__init__(
            name="Model Ensemble Specialist Agent",
            role="Select and execute predictive and diagnostic models on clean business data",
        )

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cleaned_df: pd.DataFrame = context.get("cleaned_df")
        profile: DatasetProfile = context.get("dataset_profile")
        if cleaned_df is None or profile is None:
            raise ValueError("Cleaned data or profile missing in context.")

        model_results: Dict[str, ModelResult] = {}

        # 1. Time Series Forecaster
        if profile.detected_datetime_col and profile.detected_target_col:
            forecaster = TimeSeriesForecaster(profile, horizon_days=30)
            res_fc = forecaster.run(cleaned_df)
            model_results[ModelType.TIME_SERIES.value] = res_fc

        # 2. Customer Segmentation Engine
        if profile.detected_customer_id_col:
            segmenter = CustomerSegmentationEngine(profile)
            res_seg = segmenter.run(cleaned_df)
            model_results[ModelType.CUSTOMER_SEGMENTATION.value] = res_seg

        # 3. Churn Prediction Engine
        churner = ChurnPredictionEngine(profile)
        res_churn = churner.run(cleaned_df)
        if res_churn.status != "skipped":
            model_results[ModelType.CHURN_PREDICTION.value] = res_churn

        # 4. Price Elasticity Engine
        elasticity_eng = PriceElasticityEngine(profile)
        res_elast = elasticity_eng.run(cleaned_df)
        if res_elast.status != "skipped":
            model_results[ModelType.PRICE_ELASTICITY.value] = res_elast

        # 5. Anomaly Radar Engine
        anomaly_eng = AnomalyDetectionEngine(profile)
        res_anom = anomaly_eng.run(cleaned_df)
        if res_anom.status != "skipped":
            model_results[ModelType.ANOMALY_DETECTION.value] = res_anom

        context["model_results"] = model_results
        return context
