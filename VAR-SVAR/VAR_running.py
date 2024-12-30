from dataVAR_LAGselection import optimal_lag
from dataVAR_LAGselection import data_for_var_c
from statsmodels.tsa.api import VAR

# Determine the split point (70% for training)
train_size = int(0.7 * len(data_for_var_c))

# Split the data into training and testing sets
train_data = data_for_var_c.iloc[:train_size]
test_data = data_for_var_c.iloc[train_size:]

# Train the VAR model on the training data
model = VAR(train_data)

# Select the optimal lag based on AIC (can also use BIC)
lag_order = model.select_order(maxlags=15)  # Optional maxlags for search
optimal_lag = lag_order.aic  # Selecting the lag based on AIC (you can also use BIC or FPE)


# Fit the model with the optimal lag
var_model = model.fit(2)

# Summary of the model
print(var_model.summary())




