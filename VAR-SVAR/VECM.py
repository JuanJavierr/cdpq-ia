from statsmodels.tsa.api import VECM
#from utilss import arnaud_get_datareal
from utilss import arnaud_get_data, load_data
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
split_date = '2015-12-31'

# Split the data into training and testing sets
train_data = data.loc[:split_date]  # Training data up to December 31, 2015
test_data = data.loc[split_date:]  # Testing data from January 1, 2016, onwards

from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

# Determine the cointegration rank using the Johansen test
coint_rank = select_coint_rank(train_data, det_order=0, k_ar_diff=15)  # k_ar_diff represents lags
print(coint_rank.summary())

# Define and fit the VECM model
model = VECM(
    train_data,
    k_ar_diff= 5,  # Using the cointegration rank determined above
    deterministic="ci")  # 'ci' adds intercept to the cointegrating equations

# Fit the model
vecm_model = model.fit()

print(vecm_model.summary())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import VECM

# Create an empty array to store the forecasts
forecasts = np.empty(len(end_dates))

# Iterate over each end_date in end_dates
for t in range(len(end_dates)):
    end_date = end_dates[t]

    # Select the training data up to the current end_date (this assumes 'long_run_data' is a DataFrame)
    timeseries_train = long_run_data.loc[:end_date, ['US_TB_YIELD_10YRS', 'Other_Variable1', 'Other_Variable2']]

    # If it's the first forecast, use the last 5 values
    if t == 0:
        last_train_values = timeseries_train.values[-5:]
    else:
        # For subsequent forecasts, use the last 4 values and the previous forecast
        last_train_values = np.vstack([timeseries_train.values[-4:], forecast])

    # Fit the VECM model with your parameters (adjust k_ar_diff and coint_rank as needed)
    model = VECM(timeseries_train, deterministic={"ci"}, k_ar_diff=2, coint_rank=1)
    vecm_res = model.fit()

    # Forecast 1 step ahead using the last 5 values (or 4 values + previous forecast)
    forecast = vecm_res.forecast(last_train_values, steps=1)

    # Store the forecasted value for the US_TB_YIELD_10YRS variable (adjust the index based on the structure of your forecast)
    forecasts[t] = forecast[0, 0]  # Assuming the first column corresponds to 'US_TB_YIELD_10YRS'

# Now, you can use the forecasts array for further analysis or plotting

# Example of plotting the forecast
plt.figure(figsize=(10, 6))
plt.plot(end_dates, forecasts, label='Forecasted US_TB_YIELD_10YRS', color='red', linestyle='--')
plt.title("Forecasted US_TB_YIELD_10YRS Over Time")
plt.xlabel("Time")
plt.ylabel("US_TB_YIELD_10YRS")
plt.legend()
plt.grid(True)
plt.show()

last_train_val = train_data.values[-5:]

# Initialize an empty list to store forecasted values
forecast_values = []

# Perform dynamic forecasting with actual values
#for step in range(len(test_data)):

forecast_one_step = vecm_model.predict(steps = 93)



    #forecast_values.append(forecast_one_step[0])


    #actual_next_step = test_data.iloc[step].values  # Get the actual value for the next step
    #last_train_val = np.vstack([last_train_val[1:], forecast_one_step])  # Update with actual data
# Convert forecast values to a DataFrame
forecast_df = pd.DataFrame(forecast_one_step, columns=train_data.columns)

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
