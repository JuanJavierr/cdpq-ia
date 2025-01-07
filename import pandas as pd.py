import pandas as pd
from statsmodels.tsa.stattools import adfuller
import utils

# Load data
df = utils.load_data()
  # Import the function from load_data.py




def adf_test(series):
    series = series.dropna()  # Drop NaN values before running ADF test
    result = adfuller(series, regression='n')
    adf_statistic = result[0]
    critical_values = result[4]

    print(f"ADF Statistic: {adf_statistic}")
    print(f"Critical Values: {critical_values}")

    # Compare the ADF statistic with critical values
    for key, value in critical_values.items():
        if adf_statistic < value:
            print(f"At the {key} level, the series is stationary (reject H0).")
        else:
            print(f"At the {key} level, the series is non-stationary (fail to reject H0).")


def adf_test_c(series):
    series = series.dropna()  # Drop NaN values before running ADF test
    result = adfuller(series, regression='c')
    adf_statistic = result[0]
    critical_values = result[4]

    print(f"ADF Statistic: {adf_statistic}")
    print(f"Critical Values: {critical_values}")

    # Compare the ADF statistic with critical values
    for key, value in critical_values.items():
        if adf_statistic < value:
            print(f"At the {key} level, the series is c stationary (reject H0).")
        else:
            print(f"At the {key} level, the series is c non-stationary (fail to reject H0).")


def adf_test_ct(series):
    series = series.dropna()  # Drop NaN values before running ADF test
    result = adfuller(series, regression='ct')
    adf_statistic = result[0]
    critical_values = result[4]

    print(f"ADF Statistic: {adf_statistic}")
    print(f"Critical Values: {critical_values}")

    # Compare the ADF statistic with critical values
    for key, value in critical_values.items():
        if adf_statistic < value:
            print(f"At the {key} level, the series is ct stationary (reject H0).")
        else:
            print(f"At the {key} level, the series is ct non-stationary (fail to reject H0).")

adf_test_ct(df['US_TB_YIELD_10YRS'])
adf_test_ct(df["US_TB_YIELD_1YR"])
adf_test_ct(df["US_TB_YIELD_2YRS"])

adf_test_ct(df["US_TB_YIELD_5YRS"])
adf_test_ct(df["US_TB_YIELD_3YRS"])
adf_test_ct(df["US_TB_YIELD_3MTHS"])
adf_test_ct(df["US_PERSONAL_SPENDING_PCE"])
adf_test_c(df["STICKCPIM157SFRBATL"])
adf_test_c(df["MICH"])
adf_test_ct(df["FFED"])
adf_test_ct(df["EXPINF10YR"])
adf_test_c(df["AWHMAN"])
adf_test_c(df["STDSL"])
adf_test_ct(df["SNP_500"])
adf_test_ct(df["US_CPI"])
adf_test(df["NEWS_SENTIMENT"])
adf_test_c(df["YIELD_CURVE"])
adf_test_c(df["US_UNEMPLOYMENT_RATE"])



