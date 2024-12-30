import pandas as pd
from statsmodels.tsa.stattools import adfuller
from load_ import load_and_process_data  # Import the function from load_data.py
import numpy as np
import pandas as pd


# File path to the data (update the path to your actual file)
file_path = 'Book1.xlsx'

# Load the data
monthly_avg = load_and_process_data(file_path)


def adf_test(series):
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






monthly_avg['US_CPI_diff'] = monthly_avg['US_CPI'].diff().dropna()
monthly_avg['US_PERSONAL_SPENDING_PCE_diff'] = monthly_avg['US_PERSONAL_SPENDING_PCE'].diff().dropna()
monthly_avg['SNP_500_diff'] = monthly_avg['SNP_500'].diff().dropna()
#monthly_avg['FFED_diff'] = monthly_avg['FFED'].diff().dropna()
#monthly_avg['US_TB_YIELD_10YRS_diff'] = monthly_avg['US_TB_YIELD_10YRS'].diff().dropna()
#monthly_avg['US_TB_YIELD_1YR_diff'] = monthly_avg['US_TB_YIELD_1YR'].diff().dropna()

monthly_avg['FFED_diff'] = np.log(monthly_avg['FFED'])
monthly_avg['US_TB_YIELD_10YRS_diff'] = np.log(monthly_avg['US_TB_YIELD_10YRS'])
monthly_avg['US_TB_YIELD_1YR_diff'] = np.log(monthly_avg['US_TB_YIELD_1YR'])


adf_test_c(monthly_avg["FFED"])
adf_test_c(monthly_avg["US_TB_YIELD_1YR"])
adf_test_c(monthly_avg["US_TB_YIELD_10YRS"])

adf_test_c(monthly_avg['US_CPI_diff'].dropna())
adf_test(monthly_avg['US_PERSONAL_SPENDING_PCE_diff'].dropna())
adf_test(monthly_avg['SNP_500_diff'].dropna())
adf_test(monthly_avg['FFED_diff'].dropna())
adf_test(monthly_avg['US_TB_YIELD_10YRS_diff'].dropna())
adf_test(monthly_avg['US_TB_YIELD_1YR_diff'].dropna())






