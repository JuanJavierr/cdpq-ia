import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from load_ import load_and_process_data
import statsmodels.api as sm
from statsmodels.tsa.api import VAR


file_path = 'Book1.xlsx'

monthly_avg = load_and_process_data(file_path)

monthly_avg['US_CPI_diff'] = monthly_avg['US_CPI'].diff().dropna()
monthly_avg['US_PERSONAL_SPENDING_PCE_diff'] = monthly_avg['US_PERSONAL_SPENDING_PCE'].diff().dropna()
monthly_avg['SNP_500_diff'] = monthly_avg['SNP_500'].diff().dropna()
monthly_avg['FFED_diff'] = monthly_avg['FFED'].diff().dropna()
monthly_avg['US_TB_YIELD_10YRS_diff'] = monthly_avg['US_TB_YIELD_10YRS'].diff().dropna()
monthly_avg['US_TB_YIELD_1YR_diff'] = monthly_avg['US_TB_YIELD_1YR'].diff().dropna()

# Select only the necessary columns for the VAR model
data_for = monthly_avg[['US_CPI_diff', 'US_UNEMPLOYMENT_RATE', 'US_PERSONAL_SPENDING_PCE_diff',
                        'SNP_500_diff', 'FFED_diff', 'US_TB_YIELD_10YRS_diff', 'US_TB_YIELD_10YRS_diff']]
data_for_var = data_for.dropna()

# Determine the split point (70% for training)
train_size = int(0.7 * len(data_for_var))

# Split the data into training and testing sets
train = data_for_var.iloc[:train_size]
test = data_for_var.iloc[train_size:]

# Train the VAR model on the training data
model = VAR(train)



var_model = model.fit(6)


last_train_values = train.values[-6:]

# Initialize an empty list to store forecasted values
forecast_values = []

# Perform dynamic forecasting with actual values
for step in range(len(test)):
    # Forecast one step ahead
    # Forecast one step ahead
    forecast_one_step = var_model.forecast(last_train_values, steps=12)

    # Append the forecasted value for the current step
    forecast_values.append(forecast_one_step[0])

    # Update 'last_train_values' to include the forecasted value for the next iteration
    last_train_values = np.vstack([last_train_values[1:], forecast_one_step])  # Shift data and add forecasted value
# Convert forecast values to a DataFrame
forecast_df = pd.DataFrame(forecast_values, columns=train.columns)

# Ensure forecast_df has the same index as test_data
forecast_df.index = test.index

# Convert PeriodIndex to datetime if needed
forecast_df.index = forecast_df.index.to_timestamp()  # Ensure that the forecast index is datetime
test.index = test.index.to_timestamp()  # Convert test_data index to datetime as well

# Plot the actual vs. forecasted data
plt.figure(figsize=(15, 10))

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



# Shift the test index to match the original data's index
test.index = test.index + pd.DateOffset(months=1)  # Shift test index by 1 month to match

# Initialize an empty list to store real (reconstructed) values
real_forecast_values = []

# Start with the last known real (original) values from the training data
# Get the last real value from monthly_avg using the corresponding timestamp of the last training data index
last_real_values = monthly_avg.loc[train.index[-1]].copy()


# Iterate through forecasted differenced values
for i in range(len(forecast_df)):
    # Reconstruct each differenced variable by adding the forecasted diff to the previous real value
    new_row = []
    new_row = last_real_values['US_TB_YIELD_10YRS']
    new_row = new_row + forecast_df.iloc[i]['US_TB_YIELD_10YRS_diff']

    # Append the reconstructed row to the list
    real_forecast_values.append(new_row)



# Convert the list of reconstructed values to a DataFrame
real_forecast_df = pd.DataFrame(real_forecast_values)

# Ensure the index aligns with the forecast period
real_forecast_df.index = forecast_df.index

# Print the reconstructed DataFrame
print(real_forecast_df.head())

import matplotlib.pyplot as plt

# Assuming 'real_values' contains the real values and 'forecast_df' contains the forecasted values
import matplotlib.pyplot as plt

# Assuming 'last_real_values' contains the real values and 'forecast_df' contains the forecasted values

# Make sure the indices are aligned correctly
last_real_values.index = test.index  # Aligning with 'test' DataFrame's index
forecast_df.index = forecast_df.index  # Ensure 'forecast_df' already has the correct DatetimeIndex

# Create a plot for each variable
variables_to_plot = ['US_TB_YIELD_10YRS']

plt.figure(figsize=(12, len(variables_to_plot)))

for i, var in enumerate(variables_to_plot):
    plt.subplot(len(variables_to_plot), 1, i + 1)

    # Plot real values
    plt.plot(last_real_values[var].index, last_real_values[var], label=f'Real {var}', color='blue', marker='o')

    # Plot forecasted values
    plt.plot(forecast_df.index, forecast_df[var], label=f'Forecasted {var}', color='orange', linestyle='--', marker='x')

    plt.title(f'Comparison of Real vs Forecasted {var}')
    plt.xlabel('Month')
    plt.ylabel(var)
    plt.legend()
    plt.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()



# Convert forecast_values to a NumPy array if it is not already
forecast_value = forecast_df['US_TB_YIELD_10YRS'].values


# Extract the forecasted values for 'US_TB_YIELD_10YRS'
forecasted = forecast_value


# Calculate MSE, RMSE, and MAPE
actual = test_data['US_TB_YIELD_10YRS'].values  # Replace 'US_TB_YIELD_10YRS' with your variable name

mse = np.mean((actual - forecasted) ** 2)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actual - forecasted) / actual)) * 100

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")

