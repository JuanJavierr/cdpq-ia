import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from dataVAR_LAGselection import data_for_var_c  # Assuming your data is stored in data_for_var_c
from dataVAR_LAGselection import var_model  # Assuming var_model_aic is defined in VAR_running


# Determine the split point (70% for training)
train_size = int(0.7 * len(data_for_var_c))

# Split the data into training and testing sets
train_data = data_for_var_c.iloc[:train_size]
test_data = data_for_var_c.iloc[train_size:]



# Get the last 'optimal_lag' rows from the training data to start forecasting
last_train_values = train_data.values[-6:]

forecast_values = []

# Perform recursive forecasting: Forecast one step at a time and update the training data
for step in range(len(test_data)):
    # Forecast one step ahead
    forecast_one_step = var_model.forecast(last_train_values, steps=1)


    # Append the forecasted value for the current step
    forecast_values.append(forecast_one_step[0])

    # Update 'last_train_values' to include the forecasted value for the next iteration
    last_train_values = np.vstack([last_train_values[1:], forecast_one_step])  # Shift data and add forecasted value


# Convert forecast values to DataFrame
forecast_df = pd.DataFrame(forecast_values, columns=train_data.columns)

# Ensure that forecast_df has the correct PeriodIndex or DatetimeIndex (matching test_data)
forecast_df.index = test_data.index  # Use the same index as test_data (which should be PeriodIndex)

# Convert PeriodIndex to datetime if needed (ensure that both indexes are datetime)
forecast_df.index = forecast_df.index.to_timestamp()  # Convert forecast index to datetime
test_data.index = test_data.index.to_timestamp()  # Convert test_data index to datetime

# Plot the actual vs. forecasted data
plt.figure(figsize=(15, 10))

# Plot each variable
for i, column in enumerate(test_data.columns, 1):
    plt.subplot(4, 2, i)  # Adjust subplot grid based on the number of variables
    plt.plot(test_data.index, test_data[column], label='Actual', color='blue')
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
