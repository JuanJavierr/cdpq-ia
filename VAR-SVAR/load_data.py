import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def load_and_process_data(file_path):

    # Read the Excel file (replace 'your_file.xlsx' with the actual file path)
    df = pd.read_excel('Book1.xlsx', engine='openpyxl')

    df['DATE'] = pd.to_datetime(df['DATE'])

    # Extract the month and year from the 'DATE' column
    df['Month'] = df['DATE'].dt.to_period('M')

    monthly_avg = df.groupby('Month')[['US_CPI', 'US_PERSONAL_SPENDING_PCE', 'US_UNEMPLOYMENT_RATE',
                                       'SNP_500', 'FFED', 'US_TB_YIELD_10YRS', 'US_TB_YIELD_1YR']].mean().reset_index()

    monthly_avg['Month'] = pd.to_datetime(monthly_avg['Month'].astype(str))  # Ensure datetime format
    monthly_avg.set_index('Month', inplace=True)  # Set it as the index
    monthly_avg.index = monthly_avg.index.to_period('M')  # Add frequency as monthly

    return monthly_avg

monthly_avg['US_CPI']
monthly_avg['US_PERSONAL_SPENDING_PCE']
monthly_avg['US_UNEMPLOYMENT_RATE']
monthly_avg['SNP_500']
monthly_avg['FFED']
adf_test(monthly_avg['US_TB_YIELD_10YRS'])
adf_test(monthly_avg['US_TB_YIELD_1YR'])


