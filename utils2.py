import re
import json
import pandas as pd
import numpy as np
import plotly.express as px
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import GlobalForecastingModel
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import (
    Scaler,
    MissingValuesFiller,
    Mapper,
    InvertibleMapper,
    Diff
)
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import matplotlib.pyplot as plt


def lag_monthly_macro_variables(df):
    """
    Macro-economic metrics such as CPI, PCE and unemployment rate are only known 1 month after the period they cover.
    Example: On January 15th 2016, we want to produce forecasts for February 15th 2016, but
    we only have CPI data up to December 2015.

    For this reason, we lag historical macro-economic variables by 1 prior to modeling
    """
    # print("Before shift:")
    # print(df.head())
    for col in ["US_CPI", "US_UNEMPLOYMENT_RATE", "US_PERSONAL_SPENDING_PCE"]:
        df[col] = df[col].shift(1)

    # print("After shift:")
    # print(df.head())

    return df


def load_data():
    """Load raw data and construct DataFrame with all **unscaled** features"""

    # Load SF FED data
    sf_df = pd.read_excel(Path(__file__).parent / "data/sf_fed/news_sentiment_data.xlsx", sheet_name="Data")
    sf_df = sf_df.set_index("date").asfreq("B").resample("ME").mean()
    sf_df = sf_df.rolling(window=12).mean().dropna() # Smooth data


    # Load macro-economic data
    df = pd.read_csv(Path(__file__).parent / "data/data_concours.csv", index_col=0)

    # Set date index
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").asfreq("B")

    # Keep only relevant variables
    variables = ["FFED", "US_PERSONAL_SPENDING_PCE", "US_CPI", "US_TB_YIELD_10YRS", "US_TB_YIELD_2YRS", "US_TB_YIELD_3YRS", "US_TB_YIELD_5YRS", "US_TB_YIELD_3MTHS",
                 "US_UNEMPLOYMENT_RATE", "SNP_500"]
    df = df[variables]

    # Resample to monthly frequency
    df = df.resample("ME").mean()

    # Keep last year for testing
    df = df[df.index <= "2023-08-31"]

    # Merge with SF FED data
    df = df.merge(sf_df, left_index=True, right_index=True, how="left").rename(columns={"News Sentiment": "NEWS_SENTIMENT"})

    # Keep only data from 1980 onwards
    df = df[df.index >= "1980-01-01"]

    # Lag macro-economic variables
    df = lag_monthly_macro_variables(df)

    df = df.astype(np.float32)

    return df


def scale_ts(series, should_diff, diff_order=1):
    """Scale TimeSeries and apply transformations"""
    log_transformer = InvertibleMapper(
        fn=np.log1p, inverse_fn=np.expm1, name="log1p"
    )
    scaler = Scaler(StandardScaler())
    filler = MissingValuesFiller()
    differentiator = Diff(dropna=True, lags=diff_order)

    if should_diff:
        pipeline = Pipeline([filler, differentiator, scaler])
        series_scaled = pipeline.fit_transform(series)
    else:
        pipeline = Pipeline([filler, scaler])
        series_scaled = pipeline.fit_transform(series)

    return pipeline, series_scaled


def unscale_series(series: TimeSeries, pipeline: Pipeline, ts_scaled):
    series_start_time = series.start_time()
    full_history = ts_scaled.drop_after(series_start_time).append(series)

    unscaled_full = pipeline.inverse_transform(full_history, partial=True)

    idx_start_time = unscaled_full.get_index_at_point(series_start_time)
    unscaled = unscaled_full.drop_before(idx_start_time - 1)

    return unscaled



def df2ts(df):
    # Create a TimeSeries object
    ts = TimeSeries.from_dataframe(df, value_cols=['US_TB_YIELD_10YRS'])

    # Create covariates that will be differenced
    covars_diff = df[["FFED", "US_TB_YIELD_2YRS", "US_TB_YIELD_5YRS", "US_TB_YIELD_3YRS", "US_CPI", "US_PERSONAL_SPENDING_PCE"]]
    covars_diff = TimeSeries.from_dataframe(covars_diff)

    covars_diff_yoy = df[["US_UNEMPLOYMENT_RATE", "SNP_500"]]
    covars_diff_yoy = TimeSeries.from_dataframe(covars_diff_yoy)

    # Create covariates that will not be differenced
    covars_nodiff = df[["NEWS_SENTIMENT"]]
    covars_nodiff = TimeSeries.from_dataframe(covars_nodiff)

    return ts, covars_diff, covars_diff_yoy, covars_nodiff