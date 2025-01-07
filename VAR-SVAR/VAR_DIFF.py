from statsmodels.tsa.api import VAR
#from utilss import arnaud_get_datareal
from utilss import arnaud_get_data_diff, load_data
from utilss import get_raw_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from utilss import evaluate_by_horizon

datareal = load_data()
data = arnaud_get_data_diff()
# Determine the split point (70% for training)

# Ensure the index is a DatetimeIndex
data.index = pd.to_datetime(data.index)

# Specify the date for splitting the data
split_date = '2015-12-31'

# Split the data into training and testing sets
train_data = data.loc[:split_date]  # Training data up to December 31, 2015
test_data = data.loc[split_date:]  # Testing data from January 1, 2016, onwards


# Train the VAR model on the training data
model = VAR(train_data)

# Select the optimal lag based on AIC (can also use BIC)
lag_order = model.select_order(maxlags=15)  # Optional maxlags for search
optimal_lag = lag_order.aic  # Selecting the lag based on AIC (you can also use BIC or FPE)
print(lag_order.summary())

# Fit the model with the optimal lag
var_model = model.fit(6)


print(var_model.summary())


last_train_val = train_data.values[-6:]

# Initialize an empty list to store forecasted values
forecast_values = []

# Perform dynamic forecasting with actual values
for step in range(len(test_data)):

    forecast_one_step = var_model.forecast(last_train_val, steps=1)



    forecast_values.append(forecast_one_step[0])


    #actual_next_step = test_data.iloc[step].values  # Get the actual value for the next step
    last_train_val = np.vstack([last_train_val[1:], forecast_values])  # Update with actual data
# Convert forecast values to a DataFrame
forecast_df = pd.DataFrame(forecast_values, columns=train_data.columns)

# Ensure forecast_df has the same index as test_data
forecast_df.index = test_data.index



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

plt.show()



import numpy as np


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
