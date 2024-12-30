import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def load_and_process_data(file_path):

    # Read the Excel file (replace 'your_file.xlsx' with the actual file path)
    df = pd.read_excel(file_path, engine='openpyxl')  # Use the file_path argument here

    df['DATE'] = pd.to_datetime(df['DATE'])

    # Extract the month and year from the 'DATE' column
    df['Month'] = df['DATE'].dt.to_period('M')

    # Group by 'Month' and calculate the average for the other columns
    monthly_avg = df.groupby('Month')[['US_CPI', 'US_PERSONAL_SPENDING_PCE', 'US_UNEMPLOYMENT_RATE',
                                       'SNP_500', 'FFED', 'US_TB_YIELD_10YRS', 'US_TB_YIELD_1YR']].mean().reset_index()

    # Ensure that 'Month' is in a proper datetime format (string first, then to Period)
    monthly_avg['Month'] = pd.to_datetime(monthly_avg['Month'].astype(str))

    # Set 'Month' as the index and convert to PeriodIndex with monthly frequency
    monthly_avg.set_index('Month', inplace=True)  # Set 'Month' as the index
    monthly_avg.index = monthly_avg.index.to_period('M')  # Ensure frequency is monthly

    return monthly_avg

