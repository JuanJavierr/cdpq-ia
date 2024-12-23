import re
import json
import pandas as pd
import numpy as np
import plotly.express as px
from darts import TimeSeries


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
    with open("data_concours_feature_descriptions.txt") as dd:
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


def load_data():
    df = pd.read_csv("./data_concours.csv", index_col=0)

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").asfreq("B")
    # df = df.fillna(method="ffill")

    variables = ["FFED", "US_PERSONAL_SPENDING_PCE", "US_CPI", "US_TB_YIELD_10YRS", "US_UNEMPLOYMENT_RATE"]
    df = df[variables]

    df = df.resample("ME").mean()

    # df = df.dropna()
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



def df2ts(df):
    # Create a TimeSeries object
    ts = TimeSeries.from_dataframe(df, value_cols=['US_TB_YIELD_10YRS']) #.add_holidays("US")

    # Create covariates
    covariates = df.drop(columns=['US_TB_YIELD_10YRS'])
    covariates = TimeSeries.from_dataframe(covariates)

    return ts, covariates