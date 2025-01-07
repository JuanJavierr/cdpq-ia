from VAR_running import forecast_df
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
data = arnaud_get_data_diff()
# Determine the split point (70% for training)

# Ensure the index is a DatetimeIndex
data.index = pd.to_datetime(data.index)

# Specify the date for splitting the data
split_date = '2005-12-31'
end_date = '2019-12-31'

# Split the data into training and testing sets
train_data = data.loc[:split_date]  # Training data up to December 31, 2015
test_data = data.loc[split_date:end_date]  # Testing data from January 1, 2016, onwards


# Ensure the index is a DatetimeIndex
datareal.index = pd.to_datetime(datareal.index)


# Split the data into training and testing sets
train_datareal = datareal.loc[:split_date]  # Training data up to December 31, 2015
test_datareal = datareal.loc[split_date:end_date]  # Testing data from January 1, 2016, onwards

last_trainreal_value = train_datareal['US_UNEMPLOYMENT_RATE'].iloc[-1]

real_values = []

#print(last_trainreal_value)
for i in range(len(forecast_df)):


    real_val = last_trainreal_value
    real_val = real_val + test_data['US_UNEMPLOYMENT_RATE'].iloc[i]
    #print(test_datareal['US_TB_YIELD_10YRS'].iloc[i])
    real_values.append(real_val)

# Ensure real_forecast_values is a Series with the same index as forecast_df
real_forecast_series = pd.Series(test_datareal['US_UNEMPLOYMENT_RATE'], index=forecast_df.index)
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

