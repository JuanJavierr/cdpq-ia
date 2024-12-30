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
data_for = monthly_avg[[ 'US_PERSONAL_SPENDING_PCE_diff', 'US_UNEMPLOYMENT_RATE',
                            'SNP_500_diff', 'FFED_diff', 'US_TB_YIELD_10YRS_diff', 'US_TB_YIELD_1YR_diff']]
data_for_var_c = data_for.dropna()



# Determine the split point (70% for training)
train_size = int(0.7 * len(data_for_var_c))

# Split the data into training and testing sets
train_data = data_for_var_c.iloc[:train_size]
test_data = data_for_var_c.iloc[train_size:]

# Train the VAR model on the training data
model = VAR(train_data)


lag_order = model.select_order(maxlags=15)  # Test up to 15 lags
print(lag_order.summary())

# Choose the best lag based on the lowest AIC or BIC
optimal_lag = lag_order.aic  # Or you could use lag_order.bic
print(f"Optimal lag: {optimal_lag}")

var_model = model.fit(6)

print(var_model.summary())

