"""CSV template and export helpers."""
import io
from src.data.synthetic_data_generator import LogisticsDataGenerator


def generate_sample_csv():
    data_gen = LogisticsDataGenerator()
    df_routes, _ = data_gen.generate_time_series_data(num_days=14)
    sample_df = df_routes[["date", "origin_id", "dest_id", "actual_desi", "is_campaign"]].head(100)
    buffer = io.BytesIO()
    sample_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def export_df_to_csv(df):
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer
