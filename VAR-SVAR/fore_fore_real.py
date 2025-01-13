import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from dataVAR_LAGselection import data_for_var_c  # Assuming your data is stored in data_for_var_c
from VAR_running import var_model  # Assuming var_model_aic is defined in VAR_running
from VAR_running import optimal_lag  # Assuming optimal_lag is defined in VAR_running
from VAR_running import train_data  # Assuming train_data is already split
from VAR_running import test_data  # Assuming test_data is already split
from load_ import load_and_process_data


file_path = 'Book1.xlsx'

monthly_avg = load_and_process_data(file_path)



data_var = monthly_avg[['US_PERSONAL_SPENDING_PCE', 'US_UNEMPLOYMENT_RATE',
                            'SNP_500', 'FFED', 'US_TB_YIELD_10YRS', 'US_TB_YIELD_1YR']]

# Determine the split point (70% for training)
train_size = int(0.7 * len(data_var))

# Split the data into training and testing sets
train = data_var.iloc[:train_size]
test = data_var.iloc[train_size:]

# Get the last 'optimal_lag' rows from the training data to start forecasting
last_train_values = train.values[-6:]

forecast_values = []

# Perform recursive forecasting: Forecast one step at a time and update the training data
for step in range(len(test)):
    # Forecast one step ahead
    forecast_one_step = var_model.forecast(last_train_values, steps=1)


    # Append the forecasted value for the current step
    forecast_values.append(forecast_one_step[0])

    # Replace the oldest observation with the actual test data value
    actual_next_step = test_data.iloc[step].values  # Get the actual value for the next step
    last_train_values = np.vstack([last_train_values[1:], actual_next_step])  # Update with actual data


# Convert forecast values to DataFrame
forecast_df = pd.DataFrame(forecast_values, columns=train.columns)

# Ensure that forecast_df has the correct PeriodIndex or DatetimeIndex (matching test_data)
forecast_df.index = test.index  # Use the same index as test_data (which should be PeriodIndex)

# Convert PeriodIndex to datetime if needed (ensure that both indexes are datetime)
forecast_df.index = forecast_df.index.to_timestamp()  # Convert forecast index to datetime
test.index = test.index.to_timestamp()  # Convert test_data index to datetime

# Plot the actual vs. forecasted data
plt.figure(figsize=(15, 10))

# Plot each variable
for i, column in enumerate(test.columns, 1):
    plt.subplot(4, 2, i)  # Adjust subplot grid based on the number of variables
    plt.plot(test.index, test[column], label='Actual', color='blue')
    plt.plot(forecast_df.index, forecast_df[column], label='Forecast', color='red', linestyle='dashed')
    plt.title(f'{column}: Actual vs Forecast')
    plt.xlabel('Month')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)

# Adjust layout to prevent overlapping labels
plt.tight_layout()

# Show the plot
plt.show()
