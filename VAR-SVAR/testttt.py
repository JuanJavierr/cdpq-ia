import pandas as pd
from dataVAR_LAGselection import model
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from dataVAR_LAGselection import data_for_var_c
from VAR_running import var_model_aic
from VAR_running import var_model_bic
from VAR_running import train_data
from VAR_running import test_data
from VAR_running import optimal_lag



# Forecast using the trained VAR model with observed data
forecast_static = []
history = train_data.values.tolist()  # Start with training data

# Iterate through the test set for step-by-step prediction
for t in range(len(test_data)):
    # Forecast the next value based on history
    forecast = var_model_aic.forecast(y=history[-2:], steps=1)
    forecast_static.append(forecast[0])  # Save the forecast
    # Add the actual test value to history (for static forecasting)
    history.append(test_data.values[t])

# Convert to DataFrame for easier handling
forecast_static_df = pd.DataFrame(forecast_static, columns=test_data.columns, index=test_data.index)

# Plot actual vs forecasted
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for column in test_data.columns:
    plt.plot(test_data.index, test_data[column], label=f'Actual {column}')
    plt.plot(test_data.index, forecast_static_df[column], label=f'Forecast {column}', linestyle='--')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("Actual vs Forecast (Static Forecasting)")
plt.grid()
plt.show()
