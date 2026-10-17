import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Feature engineering for LightGBM and MetaLearner models.
    Creates lag/rolling features, date indicators, campaign flags, and TC encodings.
    """
    def __init__(self):
        self.tc_encoder = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["origin_id", "dest_id", "date"]).reset_index(drop=True)
        
        # Time features
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        # Unique route ID
        df["route_id"] = df["origin_id"] + "_" + df["dest_id"]
        
        # Encode transfer center IDs
        unique_tcs = sorted(list(set(df["origin_id"]).union(set(df["dest_id"]))))
        self.tc_encoder = {tc: idx for idx, tc in enumerate(unique_tcs)}
        
        df["origin_code"] = df["origin_id"].map(self.tc_encoder)
        df["dest_code"] = df["dest_id"].map(self.tc_encoder)
        
        # Create Lag & Rolling features per route
        grouped = df.groupby("route_id")["actual_desi"]
        
        df["lag_1"] = grouped.shift(1)
        df["lag_7"] = grouped.shift(7)
        df["rolling_mean_7"] = grouped.transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
        df["rolling_mean_14"] = grouped.transform(lambda x: x.shift(1).rolling(window=14, min_periods=1).mean())
        
        # Fill early-step gaps from past-only statistics (never from the target itself).
        # The first day of each route has no history and stays NaN; both LightGBM and
        # HistGradientBoosting handle missing values natively.
        df["lag_7"] = df["lag_7"].fillna(df["rolling_mean_7"])
        df["rolling_mean_14"] = df["rolling_mean_14"].fillna(df["rolling_mean_7"])
        
        return df

    def transform_single_day(self, historical_df: pd.DataFrame, target_date: str) -> pd.DataFrame:
        transformed = self.fit_transform(historical_df)
        target_dt = pd.to_datetime(target_date)
        return transformed[transformed["date"] == target_dt]
