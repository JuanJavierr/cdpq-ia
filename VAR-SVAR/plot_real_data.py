from statsmodels.tsa.api import VAR
from utilss import arnaud_get_data, get_raw_data, load_data
from utilss import arnaud_get_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from utilss import load_data

datareal = load_data()
data = get_raw_data()
# Determine the split point (70% for training)

# Ensure the index is a DatetimeIndex
datareal.index = pd.to_datetime(datareal.index)


# Plot the actual vs. forecasted data
plt.figure(figsize=(15, 10))

for i, column in enumerate(datareal.columns, 1):
    plt.subplot(4, 2, i)  # Adjust subplot grid based on the number of variables
    plt.plot(datareal.index, datareal[column], label='Actual', color='blue')
    plt.title(f'{column}: Actual vs Forecast')
    plt.xlabel('Month')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)

# Adjust layout to prevent overlapping labels
plt.tight_layout()

# Show the plot
plt.show()