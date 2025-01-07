from statsmodels.tsa.api import VAR
#from utilss import arnaud_get_datareal
from utilss import arnaud_get_data, load_data, arnaud_get_data_diff
from utilss import get_raw_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from utilss import evaluate_by_horizon

datareal = load_data()
data = arnaud_get_data()
# Determine the split point (70% for training)

# Ensure the index is a DatetimeIndex
data.index = pd.to_datetime(data.index)

# Specify the date for splitting the data
split_date = '2012-12-31'
end_date = '2019-12-31'

# Split the data into training and testing sets
train_data = data.loc[:split_date]  # Training data up to December 31, 2015
test_data = data.loc[split_date:end_date]  # Testing data from January 1, 2016, onwards



last_train_val = train_data.values[-1:]


# Initialize an empty list to store forecasted values
forecast_values = []

print(len(test_data))
    # Perform dynamic forecasting with actual values
for step in range(len(test_data)):
    forecast_one_step = last_train_val
    forecast_values.append(forecast_one_step[0])

    actual_next_step = test_data.iloc[step].values  # Get the actual value for the next step
    last_train_val = np.vstack([actual_next_step])

print(last_train_val[0])
# Convert forecast values to a DataFrame
forecast_df = pd.DataFrame(forecast_values, columns=train_data.columns)
print(forecast_values)
# Ensure forecast_df has the same index as test_data
forecast_df.index = test_data.index



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




for step in range(len(test_data)):

    if step % 3 == 0:
        forecast_one_step = last_train_val
        forecast_values.append(forecast_one_step[0])
        forecast_values.append(forecast_one_step[0])
        forecast_values.append(forecast_one_step[0])

    actual_next_step = test_data.iloc[step].values  # Get the actual value for the next step
    last_train_val = np.vstack([actual_next_step])