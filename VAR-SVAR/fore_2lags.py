import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from dataVAR_LAGselection import data_for_var_c  # Assuming your data is stored in data_for_var_c
from VAR_running import var_model  # Assuming var_model_aic is defined in VAR_running
from VAR_running import optimal_lag  # Assuming optimal_lag is defined in VAR_running
from VAR_running import train_data  # Assuming train_data is already split
from VAR_running import test_data  # Assuming test_data is already split
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR


# Get the initial set of lagged observations from the training data
last_train_values = train_data.values[-6:]

# Initialize an empty list to store forecasted values
forecast_values = []

# Perform dynamic forecasting with actual values
for step in range(len(test_data)):
    # Forecast one step ahead
    forecast_one_step = var_model.forecast(last_train_values, steps=1)

    # Append the forecasted value
    forecast_values.append(forecast_one_step[0])

    # Replace the oldest observation with the actual test data value
    actual_next_step = test_data.iloc[step].values  # Get the actual value for the next step
    last_train_values = np.vstack([last_train_values[1:], actual_next_step])  # Update with actual data

# Convert forecast values to a DataFrame
forecast_df = pd.DataFrame(forecast_values, columns=train_data.columns)

# Ensure forecast_df has the same index as test_data
forecast_df.index = test_data.index

# Convert PeriodIndex to datetime if needed
forecast_df.index = forecast_df.index.to_timestamp()  # Ensure that the forecast index is datetime
test_data.index = test_data.index.to_timestamp()  # Convert test_data index to datetime as well

# Plot the actual vs. forecasted data
plt.figure(figsize=(15, 10))

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