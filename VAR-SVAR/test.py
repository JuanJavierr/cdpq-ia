import matplotlib.pyplot as plt
import numpy as np
from load_ import load_and_process_data  # Import the function from load_data.py
file_path = 'Book1.xlsx'
monthly_avg = load_and_process_data(file_path)
monthly_avg['US_CPI']
monthly_avg['US_PERSONAL_SPENDING_PCE']
monthly_avg['US_UNEMPLOYMENT_RATE']
monthly_avg['SNP_500']
monthly_avg['FFED']
monthly_avg['US_TB_YIELD_10YRS']
monthly_avg['US_TB_YIELD_1YR']

# Check for NaN and Inf values
print(monthly_avg['US_PERSONAL_SPENDING_PCE'].isna().sum())
print(np.isinf(monthly_avg['US_PERSONAL_SPENDING_PCE']).sum())

# Check for constant values
print(monthly_avg['US_PERSONAL_SPENDING_PCE'].nunique())

# Check the length of the data
print(len(monthly_avg['US_PERSONAL_SPENDING_PCE']))

# Check the data type
print(monthly_avg['US_PERSONAL_SPENDING_PCE'].dtype)

