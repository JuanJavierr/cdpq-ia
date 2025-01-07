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


import numpy as np

# Ensure the index is a DatetimeIndex
datareal.index = pd.to_datetime(datareal.index)

# Specify the date for splitting the data
split_date = '2015-12-31'

# Split the data into training and testing sets
train_datareal = datareal.loc[:split_date]  # Training data up to December 31, 2015
test_datareal = datareal.loc[split_date:]  # Testing data from January 1, 2016, onwards

last_trainreal_value = train_datareal["US_TB_YIELD_10YRS"].iloc[-1]


real_forecast_values = []
for i in range(len(forecast_df)):
    # Reconstruct each differenced variable by adding the forecasted diff to the previous real value
    new_row = last_trainreal_value
    new_row = new_row + forecast_df.iloc[i]['US_TB_YIELD_10YRS']

    # Append the reconstructed row to the list
    real_forecast_values.append(new_row)

# Ensure real_forecast_values is a Series with the same index as forecast_df
real_forecast_series = pd.Series(real_forecast_values, index=forecast_df.index)

# Plotting
plt.figure(figsize=(10, 6))

# Plot the actual vs reconstructed forecasted data for "US_TB_YIELD_10YRS"
plt.plot(test_datareal.index, test_datareal["US_TB_YIELD_10YRS"], label='Actual', color='blue')
plt.plot(real_forecast_series.index, real_forecast_series, label='Reconstructed Forecast', color='red', linestyle='dashed')

# Add titles and labels
plt.title('US_TB_YIELD_10YRS: Actual vs Reconstructed Forecast')
plt.xlabel('Month')
plt.ylabel('US_TB_YIELD_10YRS')
plt.legend()
plt.grid(True)

plt.tight_layout()
# Show the plot
plt.show()




import numpy as np

# Convert forecast_values to a NumPy array if it is not already
forecasted = np.array(real_forecast_values)

# Retrieve the column index for 'US_TB_YIELD_10YRS' from the test data
variable_index = test_datareal.columns.get_loc('US_TB_YIELD_10YRS')

# Calculate MSE, RMSE, and MAPE
actual = test_datareal['US_TB_YIELD_10YRS'].values  # Replace 'US_TB_YIELD_10YRS' with your variable name

mse = np.mean((actual - forecasted) ** 2)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actual - forecasted) / actual)) * 100

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")





# Convert forecast_values to a NumPy array if it is not already
forecast_values = forecast_df['US_TB_YIELD_10YRS'].values


# Extract the forecasted values for 'US_TB_YIELD_10YRS'
forecasted = forecast_values


# Calculate MSE, RMSE, and MAPE
actual = test_data['US_TB_YIELD_10YRS'].values  # Replace 'US_TB_YIELD_10YRS' with your variable name

mse = np.mean((actual - forecasted) ** 2)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actual - forecasted) / actual)) * 100

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")