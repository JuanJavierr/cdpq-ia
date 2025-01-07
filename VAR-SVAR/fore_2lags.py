import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from load_ import load_and_process_data

file_path = 'Book1.xlsx'

monthly_avg = load_and_process_data(file_path)

monthly_avg['US_CPI_diff'] = monthly_avg['US_CPI'].diff().dropna()
monthly_avg['US_PERSONAL_SPENDING_PCE_diff'] = monthly_avg['US_PERSONAL_SPENDING_PCE'].diff().dropna()
monthly_avg['SNP_500_diff'] = monthly_avg['SNP_500'].diff().dropna()
monthly_avg['FFED_diff'] = monthly_avg['FFED'].diff().dropna()
monthly_avg['US_TB_YIELD_10YRS_diff'] = monthly_avg['US_TB_YIELD_10YRS'].diff().dropna()
monthly_avg['US_TB_YIELD_1YR_diff'] = monthly_avg['US_TB_YIELD_1YR'].diff().dropna()

# Select only the necessary columns for the VAR model
data_for = monthly_avg[[  'US_UNEMPLOYMENT_RATE',
                            'SNP_500_diff', 'FFED_diff', 'US_TB_YIELD_10YRS_diff', 'US_TB_YIELD_1YR_diff']]

datareal = monthly_avg[[  'US_UNEMPLOYMENT_RATE',
                            'SNP_500', 'FFED', 'US_TB_YIELD_10YRS', 'US_TB_YIELD_1YR']]

data_for_var = data_for.dropna()



# Determine the split point (70% for training)
train_size = int(0.66 * len(data_for_var))

# Split the data into training and testing sets
train = data_for_var.iloc[:train_size]
test = data_for_var.iloc[train_size:]


# Split the data into training and testing sets
trainreal = datareal.iloc[:train_size]
testreal = datareal.iloc[train_size:]


# Train the VAR model on the training data
model = VAR(train)


lag_order = model.select_order(maxlags=15)  # Test up to 15 lags
print(lag_order.summary())

# Choose the best lag based on the lowest AIC or BIC
optimal_lag = lag_order.aic  # Or you could use lag_order.bic
print(f"Optimal lag: {optimal_lag}")

var_mode = model.fit(2)

# Get the initial set of lagged observations from the training data
last_train_values = train.values[-2:]

# Initialize an empty list to store forecasted values
forecast_values = []

# Perform dynamic forecasting with actual values
for step in range(len(test)):
    # Forecast one step ahead
    forecast_one_step = var_mode.forecast(last_train_values, steps=1)

    # Append the forecasted value
    forecast_values.append(forecast_one_step[0])

    # Replace the oldest observation with the actual test data value
    actual_next_step = test.iloc[step].values  # Get the actual value for the next step
    last_train_values = np.vstack([last_train_values[1:], actual_next_step])  # Update with actual data

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


# Ensure the index is a DatetimeIndex
#datareal.index = pd.to_datetime(datareal.index)



last_trainreal_value = trainreal['US_TB_YIELD_10YRS'].iloc[-1]

real_values = []
real_val = last_trainreal_value

#print(last_trainreal_value)
for i in range(len(forecast_df)):



    real_val = real_val + test['US_TB_YIELD_10YRS_diff'].iloc[i]
    #print(test_datareal['US_TB_YIELD_10YRS'].iloc[i])
    real_values.append(real_val)

# Ensure real_forecast_values is a Series with the same index as forecast_df
real_forecast_series = pd.Series(testreal['US_TB_YIELD_10YRS'], index=forecast_df.index)
real_values_series = pd.Series(real_values, index=forecast_df.index)

# Plotting
plt.figure(figsize=(10, 6))

# Plot the actual vs reconstructed forecasted data for "US_TB_YIELD_10YRS"
plt.plot(real_values_series.index, real_values_series, label='Actual', color='blue')
plt.plot(testreal['US_TB_YIELD_10YRS'].index, testreal['US_TB_YIELD_10YRS'], label='Reconstructed Forecast', color='red', linestyle='dashed')

# Add titles and labels
plt.title('US_TB_YIELD_10YRS: Actual vs Reconstructed Forecast')
plt.xlabel('Month')
plt.ylabel('US_TB_YIELD_10YRS')
plt.legend()
plt.grid(True)

plt.tight_layout()
# Show the plot
plt.show()



# Ensure the index is a DatetimeIndex
datareal.index = pd.to_datetime(datareal.index)


# Split the data into training and testing sets
train_datareal = datareal.loc[:split_date]  # Training data up to December 31, 2015
test_datareal = datareal.loc[split_date:end_date]  # Testing data from January 1, 2016, onwards

last_trainreal_value = train_datareal['US_TB_YIELD_10YRS'].iloc[-1]


real_forecast_values = []
real_values = []

#print(last_trainreal_value)
for i in range(len(forecast_df)):

    new_row = last_trainreal_value
    new_row = new_row + forecast_df['US_TB_YIELD_10YRS'].iloc[i]
    #print(forecast_df['US_TB_YIELD_10YRS'].iloc[i])
    real_forecast_values.append(new_row)

    real_val = last_trainreal_value
    real_val = real_val + test_data['US_TB_YIELD_10YRS'].iloc[i]
    #print(test_datareal['US_TB_YIELD_10YRS'].iloc[i])
    real_values.append(real_val)

# Ensure real_forecast_values is a Series with the same index as forecast_df
real_forecast_series = pd.Series(real_forecast_values, index=forecast_df.index)
real_values_series = pd.Series(real_values, index=forecast_df.index)

# Plotting
plt.figure(figsize=(10, 6))

# Plot the actual vs reconstructed forecasted data for "US_TB_YIELD_10YRS"
plt.plot(real_values_series.index, real_values_series, label='Actual', color='blue')
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



# Calculate MSE, RMSE, and MAPE
actual = np.array(real_values)  # Replace 'US_TB_YIELD_10YRS' with your variable name

mse = np.mean((actual - forecasted) ** 2)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actual - forecasted) / actual)) * 100

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")

