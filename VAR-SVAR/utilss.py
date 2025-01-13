import re
import json
import pandas as pd
import numpy as np
import plotly.express as px
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
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

from torch.distributed.pipelining import pipeline

config = {
    "start_date_train": "1970-01-01",
    "end_date_train": "2015-12-31",
    "start_date_test": "2016-01-01",
    "end_date_test": "2022-12-31",
    "min_variable_count": 2,
    "max_variable_count": 20,
}

def load_data_dict():
    tables = []
    with open("../data/data_concours_feature_descriptions.txt") as dd:
        dd_json = json.load(dd)
        for k in dd_json.keys():
            tables.append(pd.json_normalize(dd_json[k]))

    data_dictionary = pd.concat(tables).reset_index(drop=True)
    return data_dictionary


data_dictionary = load_data_dict()

def train_test_split(df, label_col = "FFED_diff"):
    y_train = df.loc[config["start_date_train"]:config["end_date_train"], label_col]
    X_train = df.loc[config["start_date_train"]:config["end_date_train"]].drop(columns=[label_col, "FFED"])

    y_test = df.loc[config["start_date_test"]:config["end_date_test"], label_col]
    X_test = df.loc[config["start_date_test"]:config["end_date_test"]].drop(columns=[label_col, "FFED"])

    return X_train, y_train, X_test, y_test

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

    ### Load SF FED data
    sf_df = pd.read_excel("../data/sf_fed/news_sentiment_data.xlsx", sheet_name="Data")
    sf_df = sf_df.set_index("date").asfreq("B").resample("ME").mean()


    ###
    df = pd.read_csv("../data/data_concours.csv", index_col=0)

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").asfreq("B")
    # df = df.fillna(method="ffill")

    variables = ["FFED", "US_PERSONAL_SPENDING_PCE", "US_CPI", "US_TB_YIELD_10YRS", "US_UNEMPLOYMENT_RATE", "SNP_500","US_TB_YIELD_1YR" ]
    df = df[variables]

    df = df.resample("ME").mean()

    df = df[df.index <= "2023-08-31"] # Keep last year for testing



    df = df.merge(sf_df, left_index=True, right_index=True, how="left").rename(columns={"News Sentiment": "NEWS_SENTIMENT"})

    df = df[df.index >= "1980-01-01"] # Only keep data with known sentiment

    df = lag_monthly_macro_variables(df)

    df = df.astype(np.float32)

    return df


def get_feature_desc(feature: str):
    feature_clean = feature.replace("_YOY", "").replace("_QOQ", "").replace("_MOM", "").replace("_WOW", "")
    feature_clean = re.sub(r"_\d{1,2}(?:YRS|MTHS|YR)", "", feature_clean)
    desc = data_dictionary.loc[data_dictionary["FEATURE_NAME"] == feature_clean]
    if desc.empty:
        raise Exception(f"Feature {feature} not found in data dictionary")
    return desc.iloc[0]["FEATURE_DESCRIPTION"]

def print_features_not_in_data_dict(df, data_dict):
    for feature in df.columns:
        try:
            get_feature_desc(feature)
        except:
            print(f"Feature {feature} not found in data dictionary")

def plot_features(change):
    feature = change['new']
    try:
        desc = get_feature_desc(feature)
    except:
        return
    fig = px.line(df, x=df.index, y=feature, title=feature + " " + desc)
    fig.update_layout(title={'font': {'size': 10}})
    fig.show()


def plot_forecast(real_values, forecast):
    errors = forecast - real_values
    fig = px.line()
    fig.add_scatter(x=forecast.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red'))
    fig.add_scatter(x=real_values.index, y=real_values, mode='lines', name='Actual')
    fig.add_bar(x=errors.index, y=errors, name='Errors')
    fig.update_layout(title='Simple Exponential Smoothing vs Actual', xaxis_title='Date', yaxis_title='Value')
    fig.show()

def unscale_series(series: TimeSeries, pipeline: Pipeline, ts_scaled):
    series_start_time = series.start_time()
    full_history = ts_scaled.drop_after(series_start_time).append(series)

    unscaled_full = pipeline.inverse_transform(full_history, partial=True)

    idx_start_time = unscaled_full.get_index_at_point(series_start_time)
    unscaled = unscaled_full.drop_before(idx_start_time - 1)

    return unscaled
    

def make_forecasts(model: TorchForecastingModel, ts, ts_scaled, covariates_scaled, pipeline):
    # TODO: Refactor into 2 functions: make_forecasts and get_labels_for_period
    
    forecasts = pd.DataFrame()

    val_df_scaled = ts_scaled.drop_before(pd.Timestamp("2015-12-31")).pd_dataframe()

    for i, t in enumerate(val_df_scaled.index):
        ts_up_to_t = ts_scaled.drop_after(t)
        covariates = covariates_scaled.drop_after(t)

        # print(f"Producing forecasts made at date: {ts_up_to_t.end_time()}")
        # Make forecasts
        pred = model.predict(n=12, series=ts_up_to_t, past_covariates=covariates, num_samples=500)

        # Get values for each quantile and unscale
        pred_quantiles_unscaled = {q: unscale_series(pred.quantile(q), pipeline, ts_scaled).pd_series() for q in [0.05, 0.1, 0.5, 0.9, 0.95]}
        # print(pred_unscaled)
        # print(pred)

        idx = ts.get_index_at_point(t)
        labels = ts.pd_dataframe().iloc[idx:idx+12]
        # print(labels)

        labels["lowest_q"] = pred_quantiles_unscaled[0.05]
        labels["low_q"] = pred_quantiles_unscaled[0.1]
        labels["forecast"] = pred_quantiles_unscaled[0.5]
        labels["high_q"] = pred_quantiles_unscaled[0.9]
        labels["highest_q"] = pred_quantiles_unscaled[0.95]
        labels["error"] = labels["US_TB_YIELD_10YRS"] - labels["forecast"]
        labels["forecast_date"] = ts_up_to_t.end_time()

        # print(labels)
        forecasts = pd.concat([forecasts, labels])

    forecasts["horizon"] = (forecasts.index.to_period("M") - forecasts.forecast_date.dt.to_period("M")).map(lambda x: x.n)
    print(forecasts.groupby(by="horizon").mean()[["error"]])

    return forecasts

def get_ts_by_forecast_horizon(pred_df):
    forecast_by_horizon = {}
    for h in pred_df["horizon"].unique():
        fore = pred_df[pred_df["horizon"] == h]
        fore = fore.asfreq("ME")
        # forecast_by_horizon[h] = TimeSeries.from_dataframe(fore, value_cols=["forecast"])
        values = np.stack([fore["lowest_q"], fore["low_q"], fore["forecast"], fore["high_q"], fore["highest_q"]], axis=1)
        values = np.expand_dims(values, axis=1)
        forecast_by_horizon[h] = TimeSeries.from_times_and_values(times=fore.index, values=values)

    return forecast_by_horizon



def df2ts(df):
    # Create a TimeSeries object
    ts = TimeSeries.from_dataframe(df, value_cols=['US_TB_YIELD_10YRS']) #.add_holidays("US")

    # Create covariates that will be differenced
    covars_diff = df[["US_CPI", "FFED", "SNP_500", "US_PERSONAL_SPENDING_PCE", "US_TB_YIELD_1YR"]]
    covars_diff = TimeSeries.from_dataframe(covars_diff)

    # Create covariates that will not be differenced
    covars = df[["US_UNEMPLOYMENT_RATE", "NEWS_SENTIMENT"]]
    covars = TimeSeries.from_dataframe(covars)


    return ts, covars_diff, covars

def scale(series, should_diff):
    filler = MissingValuesFiller()
    differentiator = Diff(dropna=True)
    if should_diff:
            pipeline = Pipeline([filler, differentiator])
            series_scaled = pipeline.fit_transform(series)
    else:
            pipeline = Pipeline([filler])
            series_scaled = pipeline.fit_transform(series)[1:]

    return pipeline, series_scaled


def scale_ts(series, should_diff):
    #log_transformer = InvertibleMapper(
        #fn=np.log1p, inverse_fn=np.expm1, name="log1p"
    #)
    #scaler = Scaler(StandardScaler())
    filler = MissingValuesFiller()
    differentiator = Diff(dropna=True)

    if should_diff:
        pipeline = Pipeline([filler, differentiator])
        series_scaled = pipeline.fit_transform(series)
    else:
        pipeline = Pipeline([filler])
        series_scaled = pipeline.fit_transform(series)[1:]

    return pipeline, series_scaled


def get_raw_data():
    # Load the raw data using the existing load_data function
    df = load_data()

    # Convert to TimeSeries objects without any transformations
    ts, covars_diff, covars_nodiff = df2ts(df)

    # Return the untransformed data (without any scaling or differencing)
    df_raw = covars_diff.stack(covars_nodiff).stack(ts).pd_dataframe()

    return df_raw




def plot_training_history(train_losses, val_losses):
    # Create the figure
    fig = go.Figure()

    # Add traces for training and validation loss
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(train_losses) + 1)),
            y=train_losses,
            name='Training Loss',
            line=dict(color='blue')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(val_losses) + 1)),
            y=val_losses,
            name='Validation Loss',
            line=dict(color='red')
        )
    )

    # Update layout
    fig.update_layout(
        title='Training and Validation Loss Over Time',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified',
        template='plotly_white'
    )

    # Show the plot
    fig.show()


def evaluate_by_horizon(forecasts_df):
    forecasts_df["abs_pct_error"] = ((forecasts_df["US_TB_YIELD_10YRS"] - forecasts_df["forecast"]) / forecasts_df["US_TB_YIELD_10YRS"]).abs()
    forecasts_df["squared_error"] = forecasts_df["error"]**2
    grouped = forecasts_df.groupby(by="horizon").mean()[["abs_pct_error", "squared_error"]].add_prefix("mean_")
    grouped["root_mean_squared_error"] = grouped["mean_squared_error"]**0.5

    return grouped


def save_results(hparams, eval_metrics, output_path):
    output_path = Path(output_path)
    include_header = not output_path.exists()
    results = pd.concat([pd.Series(hparams), eval_metrics.mean()])
    pd.DataFrame(results).T.to_csv(output_path, mode="a", header=include_header, index=False)




def arnaud_get_data():

    df = load_data()
    ts, covars_diff, covars_nodiff = df2ts(df)

    covars_diff_pipeline, covars_diff_scaled = scale_ts(covars_diff, should_diff=False)
    covars_nodiff_pipeline, covars_nodiff_scaled = scale_ts(
        covars_nodiff, should_diff=False
    )
    pipeline, ts_scaled = scale_ts(ts, should_diff=False)


    covariates_scaled = covars_diff_scaled.stack(covars_nodiff_scaled)

    covariates_scaled = covariates_scaled.stack(ts_scaled)
    df = covariates_scaled.pd_dataframe()


    return df


def arnaud_get_data_diff():

    df = load_data()
    ts, covars_diff, covars_nodiff = df2ts(df)

    covars_diff_pipeline, covars_diff_scaled = scale(covars_diff, should_diff=True)
    covars_nodiff_pipeline, covars_nodiff_scaled = scale(covars_nodiff, should_diff=False)
    pipeline, ts_scaled = scale(ts, should_diff=True)


    covariates_scaled = covars_diff_scaled.stack(covars_nodiff_scaled)

    covariates_scaled = covariates_scaled.stack(ts_scaled)
    df = covariates_scaled.pd_dataframe()


    return df