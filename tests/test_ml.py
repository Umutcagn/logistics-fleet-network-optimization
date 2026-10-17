import pytest
import numpy as np
import pandas as pd
from src.data.synthetic_data_generator import LogisticsDataGenerator
from src.ml.feature_engineering import FeatureEngineer
from src.ml.lightgbm_model import LightGBMForecaster
from src.ml.meta_learner import StackingMetaLearner, calculate_wape

def test_wape_calculation():
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([110.0, 190.0, 300.0])
    wape = calculate_wape(actual, predicted)
    assert round(wape, 4) == round(20.0 / 600.0, 4)

def test_feature_engineer_and_lightgbm():
    gen = LogisticsDataGenerator()
    df_routes, _ = gen.generate_time_series_data(num_days=15)
    
    fe = FeatureEngineer()
    df_feat = fe.fit_transform(df_routes)
    
    assert "lag_1" in df_feat.columns
    assert "rolling_mean_7" in df_feat.columns
    assert "origin_code" in df_feat.columns
    
    model = LightGBMForecaster()
    model.fit(df_feat)
    preds = model.predict(df_feat)
    
    assert len(preds) == len(df_feat)
    assert np.all(preds >= 10.0)

def test_meta_learner_weights():
    actual = np.array([100, 200, 300, 400])
    lgbm_preds = np.array([105, 195, 310, 390])
    lstm_preds = np.array([90, 210, 280, 420])
    
    meta = StackingMetaLearner()
    meta.fit_weights(actual, lgbm_preds, lstm_preds)
    
    assert 0.0 <= meta.w1 <= 1.0
    assert 0.0 <= meta.w2 <= 1.0
    assert round(meta.w1 + meta.w2, 2) == 1.0


def _split(num_days=30):
    df_routes, _ = LogisticsDataGenerator().generate_time_series_data(num_days=num_days)
    df_feat = FeatureEngineer().fit_transform(df_routes)
    cutoff = df_feat["date"].max() - pd.Timedelta(days=5)
    return df_feat[df_feat["date"] <= cutoff], df_feat[df_feat["date"] > cutoff]


def test_lag_features_never_use_current_target():
    df_routes, _ = LogisticsDataGenerator().generate_time_series_data(num_days=15)
    df_feat = FeatureEngineer().fit_transform(df_routes)
    first_days = df_feat.groupby("route_id").head(1)
    assert first_days[["lag_1", "rolling_mean_7"]].isna().all().all()


def test_lstm_is_day_ahead_and_does_not_see_its_own_target():
    from src.ml.lstm_model import PyTorchLSTMForecaster

    df_train, df_val = _split()
    model = PyTorchLSTMForecaster(sequence_length=14, epochs=30).fit(df_train)
    preds = model.predict(df_val)
    assert len(preds) == len(df_val)

    # Changing a row's own actual value must not change its forecast
    last_day = df_val["date"] == df_val["date"].max()
    tampered = df_val.copy()
    tampered.loc[last_day, "actual_desi"] *= 10
    assert np.allclose(preds[last_day.values], model.predict(tampered)[last_day.values])


def test_lstm_beats_constant_floor():
    from src.ml.lstm_model import PyTorchLSTMForecaster

    df_train, df_val = _split()
    preds = PyTorchLSTMForecaster(sequence_length=14).fit(df_train).predict(df_val)
    assert calculate_wape(df_val["actual_desi"].values, preds) < 0.3
