"""
Time Series and Demand Forecasting Engine for SMBs.
Performs automated cadence aggregation, trend/seasonality decomposition,
ensemble forecasting, uncertainty intervals, and business-focused growth projections.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_intelligence.types import ModelType, ModelResult, DatasetProfile


class TimeSeriesForecaster:
    """
    Automated forecaster tailored for SMB operational and financial metrics.
    """

    def __init__(self, profile: DatasetProfile, horizon_days: int = 30):
        self.profile = profile
        self.horizon_days = horizon_days

    def run(self, df: pd.DataFrame, date_col: Optional[str] = None, target_col: Optional[str] = None) -> ModelResult:
        """
        Execute automated time-series forecasting pipeline.
        """
        start_time = time.time()
        d_col = date_col or self.profile.detected_datetime_col
        t_col = target_col or self.profile.detected_target_col

        if not d_col or not t_col or d_col not in df.columns or t_col not in df.columns:
            return ModelResult(
                model_type=ModelType.TIME_SERIES,
                model_name="Ensemble Trend & Seasonality Forecaster",
                status="skipped",
                insights=["Forecasting skipped: Valid datetime and numerical target columns were not identified."],
                error_message="Missing datetime or target column.",
                execution_time_seconds=round(time.time() - start_time, 2),
            )

        try:
            # 1. Prepare and aggregate time series to daily frequency
            work_df = df[[d_col, t_col]].dropna().copy()
            work_df[d_col] = pd.to_datetime(work_df[d_col], errors="coerce")
            work_df = work_df.dropna(subset=[d_col])

            daily_ts = work_df.groupby(pd.Grouper(key=d_col, freq="D"))[t_col].sum().reset_index()
            daily_ts.columns = ["date", "actual"]

            # Filter out trailing zero days if partial
            daily_ts = daily_ts[daily_ts["actual"] > 0].reset_index(drop=True)

            if len(daily_ts) < 14:
                return ModelResult(
                    model_type=ModelType.TIME_SERIES,
                    model_name="Ensemble Trend & Seasonality Forecaster",
                    status="warning",
                    insights=["Insufficient historical observations (minimum 14 daily data points required for reliable forecasting)."],
                    execution_time_seconds=round(time.time() - start_time, 2),
                )

            # 2. Feature engineering for regression + seasonal components
            daily_ts["day_idx"] = np.arange(len(daily_ts))
            daily_ts["dow"] = daily_ts["date"].dt.day_of_week
            daily_ts["is_weekend"] = daily_ts["dow"].isin([5, 6]).astype(int)
            daily_ts["month"] = daily_ts["date"].dt.month

            # One-hot encode day of week
            dow_dummies = pd.get_dummies(daily_ts["dow"], prefix="dow", drop_first=True)
            feature_cols = ["day_idx", "is_weekend"] + list(dow_dummies.columns)
            X = pd.concat([daily_ts[["day_idx", "is_weekend"]], dow_dummies], axis=1)
            y = daily_ts["actual"].values

            # 3. Model Training & Evaluation
            train_size = max(7, int(len(daily_ts) * 0.85))
            X_train, y_train = X.iloc[:train_size], y[:train_size]
            X_val, y_val = X.iloc[train_size:], y[train_size:]

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)

            # Fitted on full dataset
            model_full = Ridge(alpha=1.0)
            model_full.fit(X, y)
            in_sample_preds = np.clip(model_full.predict(X), 0, None)

            # Evaluate metrics on validation set if available
            if len(X_val) > 0:
                val_preds = np.clip(model.predict(X_val), 0, None)
                mae = float(mean_absolute_error(y_val, val_preds))
                rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
                mape = float(np.mean(np.abs((y_val - val_preds) / np.maximum(y_val, 1.0))) * 100)
                r2 = float(r2_score(y_val, val_preds))
            else:
                mae = float(mean_absolute_error(y, in_sample_preds))
                rmse = float(np.sqrt(mean_squared_error(y, in_sample_preds)))
                mape = float(np.mean(np.abs((y - in_sample_preds) / np.maximum(y, 1.0))) * 100)
                r2 = float(r2_score(y, in_sample_preds))

            # 4. Out-of-Sample Forecasting
            last_date = daily_ts["date"].max()
            last_idx = daily_ts["day_idx"].max()
            future_dates = [last_date + timedelta(days=i + 1) for i in range(self.horizon_days)]

            future_df = pd.DataFrame({"date": future_dates})
            future_df["day_idx"] = [last_idx + 1 + i for i in range(self.horizon_days)]

        return None
