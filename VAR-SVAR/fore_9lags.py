import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
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
data_for = monthly_avg[[  'US_UNEMPLOYMENT_RATE',
                            'SNP_500_diff', 'FFED_diff', 'US_TB_YIELD_10YRS_diff']]
data_for_var = data_for.dropna()



# Determine the split point (70% for training)
train_size = int(0.7 * len(data_for_var))

# Split the data into training and testing sets
train = data_for_var.iloc[:train_size]
test = data_for_var.iloc[train_size:]

# Train the VAR model on the training data
model = VAR(train)


lag_order = model.select_order(maxlags=15)  # Test up to 15 lags
print(lag_order.summary())

# Choose the best lag based on the lowest AIC or BIC
optimal_lag = lag_order.aic  # Or you could use lag_order.bic
print(f"Optimal lag: {optimal_lag}")

var_model = model.fit(6)

print(var_model.summary())




last_train_values = train.values[-6:]

# Initialize an empty list to store forecasted values
forecast_values = []

# Perform dynamic forecasting with actual values
for step in range(len(test)):
    # Forecast one step ahead
    # Forecast one step ahead
    forecast_one_step = var_model.forecast(last_train_values, steps=1)

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
