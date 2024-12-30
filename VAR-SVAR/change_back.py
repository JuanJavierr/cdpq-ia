import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
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

model = VAR(train)


var_model = model.fit(6)



# Get the last 'optimal_lag' rows from the training data to start forecasting
last_train_values = train.values[-6:]

forecast_values = []

# Perform recursive forecasting: Forecast one step at a time and update the training data
for step in range(len(test)):
    # Forecast one step ahead
    forecast_one_step = var_model.forecast(last_train_values, steps=1)

    # Append the forecasted value for the current step
    forecast_values.append(forecast_one_step[0])

    # Update 'last_train_values' to include the forecasted value for the next iteration
    last_train_values = np.vstack([last_train_values[1:], forecast_one_step])  # Shift data and add forecasted value


print(forecast_values)
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

