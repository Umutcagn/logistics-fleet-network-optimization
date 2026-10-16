import pandas as pd
import numpy as np

class LightGBMForecaster:
    """
    First Layer Forecaster: LightGBM
    Captures short-term campaign effects, day-of-week patterns, and TC encoding.
    Reference: Weng et al. 2020
    """
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'regression',
            'metric': 'mape',
            'n_estimators': 150,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42,
            'verbose': -1
        }
        self.model = None
        self.feature_cols = [
            'distance_km', 'day_of_week', 'is_weekend', 'is_campaign', 'is_holiday',
            'origin_code', 'dest_code', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14'
        ]

    def fit(self, df: pd.DataFrame):
        X = df[self.feature_cols]
        y = df['actual_desi']
        
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(X, y)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor
            print("[Warning] LightGBM not found, falling back to HistGradientBoostingRegressor.")
            self.model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=42)
            self.model.fit(X, y)
            
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("LightGBM model is not fitted yet.")
        X = df[self.feature_cols]
        preds = self.model.predict(X)
        return np.maximum(10.0, preds)  # Ensure non-negative demand
