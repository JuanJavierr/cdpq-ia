import pandas as pd
from statsmodels.tsa.stattools import adfuller
from load_ import load_and_process_data  # Import the function from load_data.py

# File path to the data (update the path to your actual file)
file_path = 'Book1.xlsx'

# Load the data
monthly_avg = load_and_process_data(file_path)


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


# Example: Test the original series with different regression assumptions
#adf_test(monthly_avg['US_CPI_MOM'])
adf_test(monthly_avg['US_PERSONAL_SPENDING_PCE_MOM'])
adf_test(monthly_avg['US_UNEMPLOYMENT_RATE_MOM'])
adf_test(monthly_avg['SNP_500_MOM'])
adf_test(monthly_avg['FFED_MOM'])
adf_test(monthly_avg['US_TB_YIELD_10YRS'])

adf_test_c(monthly_avg['US_CPI_MOM'])
adf_test_c(monthly_avg['US_PERSONAL_SPENDING_PCE_MOM'])

adf_test_ct(monthly_avg['US_TB_YIELD_10YRS'])

# Now run tests on the differenced series
monthly_avg['US_CPI_MOM_diff'] = monthly_avg['US_CPI_MOM'].diff().dropna()
monthly_avg['US_PERSONAL_SPENDING_PCE_diff'] = monthly_avg['US_PERSONAL_SPENDING_PCE_MOM'].diff().dropna()
monthly_avg['US_TB_YIELD_10YRS_diff'] = monthly_avg['US_TB_YIELD_10YRS'].diff().dropna()

adf_test_c(monthly_avg['US_CPI_MOM_diff'])
adf_test(monthly_avg['US_PERSONAL_SPENDING_PCE_diff'])
adf_test_ct(monthly_avg['US_TB_YIELD_10YRS_diff'])
