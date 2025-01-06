from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import (
    Diff,
    InvertibleMapper,
    MissingValuesFiller,
    Scaler,
)
from sklearn.preprocessing import StandardScaler


#### Deep Learning utils
def save_results(hparams, eval_metrics, output_path):
    output_path = Path(output_path)
    include_header = not output_path.exists()
    results = pd.concat([pd.Series(hparams), eval_metrics.mean()])
    pd.DataFrame(results).T.to_csv(
        output_path, mode="a", header=include_header, index=False
    )

def make_forecasts(model, ts: TimeSeries, ts_scaled: TimeSeries, covariates_scaled: TimeSeries, pipeline:Pipeline) -> pd.DataFrame:
    # TODO: Refactor into 2 functions: make_forecasts and get_labels_for_period
    
    forecasts = pd.DataFrame()

    val_df_scaled = ts_scaled.drop_before(pd.Timestamp("2015-12-31")).pd_dataframe()


    # Make forecasts for each date in the validation set
    for t in val_df_scaled.index:

        # Get data up to date t
        ts_up_to_t = ts_scaled.drop_after(t)
        if covariates_scaled is not None:
            covariates = covariates_scaled.drop_after(t)
        else:
            covariates = None

        # print(f"Producing forecasts made at date: {ts_up_to_t.end_time()}")
        if model.supports_probabilistic_prediction:
            # Make forecasts
            pred = model.predict(n=12, series=ts_up_to_t, past_covariates=covariates, num_samples=500, verbose=False)

            # Get values for each quantile and unscale
            pred_quantiles_unscaled = {q: unscale_series(pred.quantile(q), pipeline, ts_scaled).pd_series() for q in [0.05, 0.1, 0.5, 0.9, 0.95]}
            # print(pred_unscaled)
            # print(pred)
        else:
            pred = model.predict(n=12, series=ts_up_to_t, past_covariates=covariates)
            pred_unscaled = unscale_series(pred, pipeline, ts_scaled).pd_series()
            pred_quantiles_unscaled = {0.5: pred_unscaled}
            for q in [0.05, 0.1, 0.9, 0.95]:
                pred_quantiles_unscaled[q] = pred_unscaled

        # Get labels (real values) for the period
        labels = ts.pd_dataframe().loc[t:t+pd.Timedelta(days=364)]
        # print(labels)



        # Create a dataframe with the forecasted values
        labels["lowest_q"] = pred_quantiles_unscaled[0.05]
        labels["low_q"] = pred_quantiles_unscaled[0.1]
        labels["forecast"] = pred_quantiles_unscaled[0.5]
        labels["high_q"] = pred_quantiles_unscaled[0.9]
        labels["highest_q"] = pred_quantiles_unscaled[0.95]
        labels["error"] = labels["US_TB_YIELD_10YRS"] - labels["forecast"]
        labels["forecast_date"] = ts_up_to_t.end_time()

        # print(labels)
        # Append to forecasts
        forecasts = pd.concat([forecasts, labels])


    # Horizon is the number of months between the forecast date and the date of the forecast
    forecasts["horizon"] = (forecasts.index.to_period("M") - forecasts.forecast_date.dt.to_period("M")).map(lambda x: x.n)

    # Print evaluation metrics
    # print(forecasts.groupby(by="horizon").mean()[["error"]])

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

def evaluate_by_horizon(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate forecasts by horizon
    Args:
        forecasts_df: DataFrame containing forecasts and real values. Must have columns "forecast", "US_TB_YIELD_10YRS" and "horizon".
        US_TB_YIELD_10YRS is the real value. forecast is the forecasted value. horizon is the number of months between the forecast date and the date of the forecast
    """
    # Compute error and squared error
    forecasts_df["abs_pct_error"] = ((forecasts_df["US_TB_YIELD_10YRS"] - forecasts_df["forecast"]) / forecasts_df["US_TB_YIELD_10YRS"]).abs() * 100
    forecasts_df["squared_error"] = forecasts_df["error"]**2

    # Group by horizon and compute mean error and mean squared error
    grouped = forecasts_df.groupby(by="horizon").mean()[["abs_pct_error", "squared_error"]].add_prefix("mean_")
    grouped["root_mean_squared_error"] = grouped["mean_squared_error"]**0.5

    return grouped


#### Plot utils
def plot_training_history(train_losses, val_losses):
    # Create the figure
    fig = go.Figure()

    # Add traces for training and validation loss
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(train_losses) + 1)),
            y=train_losses,
            name="Training Loss",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(val_losses) + 1)),
            y=val_losses,
            name="Validation Loss",
            line=dict(color="red"),
        )
    )

    # Update layout
    fig.update_layout(
        title="Training and Validation Loss Over Time",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
        template="plotly_white",
    )

    # Show the plot
    fig.show()


def plot_forecast_for_horizon(h, forecasts_ts, axs, ts):
    figsize = (9, 6)
    lowest_q, low_q, high_q, highest_q = 0.05, 0.1, 0.9, 0.95
    label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
    label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"

    fcast = forecasts_ts[h]

    # plot actual series
    # plt.figure(figsize=figsize)
    ts[fcast.start_time() :].plot(label="actual", ax=axs)

    # plot prediction with quantile ranges
    fcast.plot(low_quantile=0.1, high_quantile=0.95, label=label_q_outer, ax=axs)
    fcast.plot(low_quantile=0.3, high_quantile=0.7, label=label_q_inner, ax=axs)

    # if axs.get_legend():
    #     axs.get_legend().remove()
    # axs.set_xlabel("")

    # plt.title(f"MAPE: {mape(ts, fcast):.2f}%")
    axs.set_title(f"{h}-month ahead forecast", fontsize=8)
    plt.legend()
    plt.show()


### Data Loading and Preprocessing
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


def load_jorge_data():
    # deux_semaines = pd.read_csv("data/excel_jorge/Variables_Chomage_US_2semaines.csv")
    # deux_semaines["DATE"] = pd.to_datetime(deux_semaines["DATE"])

    # trimestrielles = pd.read_excel("data/excel_jorge/Variables_US_Trimestrielles.xlsx")
    # weekly = pd.read_csv("data/excel_jorge/Variables_US_Weekly.csv")

    autres = pd.read_csv("data/excel_jorge/Variables_US.csv")
    autres["DATE"] = pd.to_datetime(autres["DATE"])
    autres = autres.set_index("DATE").asfreq("MS")
    autres = autres.replace("Nan", pd.NA)
    autres = autres.apply(pd.to_numeric, errors="ignore")
    autres = autres.resample("ME").mean()

    autres = autres[
        ["STICKCPIM157SFRBATL", "MICH", "AWHMAN", "EMRATIO", "STDSL", "EXPINF10YR"]
    ]

    return autres


def load_data():
    """Load raw data and construct DataFrame with all **unscaled** features"""

    # Load SF FED data
    sf_df = pd.read_excel(
        Path(__file__).parent / "data/sf_fed/news_sentiment_data.xlsx",
        sheet_name="Data",
    )
    sf_df = sf_df.set_index("date").asfreq("B").resample("ME").mean()
    sf_df = sf_df.rolling(window=12).mean().dropna()  # Smooth data

    # Load macro-economic data
    df = pd.read_csv(Path(__file__).parent / "data/data_concours.csv", index_col=0)

    # Set date index
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").asfreq("B")

    # Keep only relevant variables
    variables = [
        "FFED",
        "US_PERSONAL_SPENDING_PCE",
        "US_CPI",
        "US_TB_YIELD_10YRS",
        "US_TB_YIELD_1YR",
        "US_TB_YIELD_2YRS",
        "US_TB_YIELD_3YRS",
        "US_TB_YIELD_5YRS",
        "US_TB_YIELD_3MTHS",
        "US_UNEMPLOYMENT_RATE",
        "SNP_500",
    ]
    df = df[variables]


    # Compute yield curve indicator
    df["YIELD_CURVE"] = df["US_TB_YIELD_10YRS"] - df["US_TB_YIELD_3MTHS"]


    # Resample to monthly frequency
    df = df.resample("ME").mean()

    # Keep last year for testing
    df = df[df.index <= "2023-08-31"]



    ###### Add Jorge's data
    jorge_df = load_jorge_data()
    df = df.merge(jorge_df, left_index=True, right_index=True, how="left")


    # Merge with SF FED data
    df = df.merge(sf_df, left_index=True, right_index=True, how="left").rename(
        columns={"News Sentiment": "NEWS_SENTIMENT"}
    )

    # Keep only data from 1980 onwards
    df = df[df.index >= "1980-01-01"]

    # Lag macro-economic variables
    df = lag_monthly_macro_variables(df)

    df = df.astype(np.float32)

    return df


def scale_ts(series, should_diff, diff_order=1):
    """Scale TimeSeries and apply transformations"""
    log_transformer = InvertibleMapper(fn=np.log1p, inverse_fn=np.expm1, name="log1p")
    scaler = Scaler(StandardScaler())
    filler = MissingValuesFiller()
    differentiator = Diff(dropna=True, lags=diff_order)

    if should_diff:
        pipeline = Pipeline([filler, log_transformer, differentiator, scaler])
        series_scaled = pipeline.fit_transform(series)
    else:
        pipeline = Pipeline([filler, log_transformer, scaler])
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
    ts = TimeSeries.from_dataframe(df, value_cols=["US_TB_YIELD_10YRS"])

    varss = [
            "FFED",
            "US_TB_YIELD_1YR",
            "US_TB_YIELD_2YRS",
            "US_TB_YIELD_5YRS",
            "US_TB_YIELD_3YRS",
            "US_TB_YIELD_5YRS",
            "US_TB_YIELD_3MTHS",
            "US_PERSONAL_SPENDING_PCE",
            "STICKCPIM157SFRBATL",
            "MICH", # EXPECTED INFLATION 1 YR
            "EXPINF10YR", # EXPECTED INFLATION 10 YR
            "AWHMAN",
            "STDSL", # SMALL DEPOSITS,

            #### YOY
            "SNP_500", # STOCK MARKET
            "US_CPI", # INFLATION


            # NO DIFF
            "NEWS_SENTIMENT",
            "YIELD_CURVE",
            "US_UNEMPLOYMENT_RATE",
            "STDSL"
        ]

    # Create covariates that will be differenced
    covars_diff = df[varss]
    covars_diff.columns = [col + "_diff" for col in covars_diff.columns]
    covars_diff = TimeSeries.from_dataframe(covars_diff)


    covars_diff_yoy = df[varss]
    covars_diff_yoy.columns = [col + "_yoy" for col in covars_diff_yoy.columns]
    covars_diff_yoy = TimeSeries.from_dataframe(covars_diff_yoy)

    # Create covariates that will not be differenced
    covars_nodiff = df[varss]
    covars_nodiff.columns = [col + "_nodiff" for col in covars_nodiff.columns]
    covars_nodiff = TimeSeries.from_dataframe(covars_nodiff)



    covars_diff_qoq = df[varss]
    # Append qoq_ before the column names
    covars_diff_qoq.columns = [col + "_qoq" for col in covars_nodiff.columns]
    covars_diff_qoq = TimeSeries.from_dataframe(covars_diff_qoq)

    return ts, covars_diff, covars_diff_yoy, covars_nodiff, covars_diff_qoq
