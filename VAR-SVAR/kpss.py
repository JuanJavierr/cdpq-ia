import pandas as pd
from statsmodels.tsa.stattools import kpss
from load_data import load_and_process_data  # Import the function from load_data.py

# File path to the data (update the path to your actual file)
file_path = 'Book1.xlsx'

# Load the data
monthly_avg = load_and_process_data(file_path)


def kpss_test(series):
    result = kpss(series, regression='c')  # 'c' for constant (mean non-zero), 't' for trend
    print(f"KPSS Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"Critical Values: {result[3]}")

    if result[1] <= 0.05:
        print("The series is likely non-stationary (reject H0)")
    else:
        print("The series is likely stationary (fail to reject H0)")

