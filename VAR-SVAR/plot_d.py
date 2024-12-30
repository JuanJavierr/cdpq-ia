import matplotlib.pyplot as plt
from load_ import load_and_process_data  # Import the function from load_data.py

# File path to the data (update the path to your actual file)
file_path = 'Book1.xlsx'

# Load the data
monthly_avg = load_and_process_data(file_path)

# Compute the differences
monthly_avg['US_CPI_MOM_diff'] = monthly_avg['US_CPI_MOM'].diff().dropna()
monthly_avg['US_PERSONAL_SPENDING_PCE_diff'] = monthly_avg['US_PERSONAL_SPENDING_PCE_MOM'].diff().dropna()
monthly_avg['US_TB_YIELD_10YRS_diff'] = monthly_avg['US_TB_YIELD_10YRS'].diff().dropna()

# Create individual plots
plt.figure(figsize=(10, 8))

# Plot for 'US_CPI_MOM_diff'
plt.subplot(3, 2, 1)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['US_CPI_MOM_diff'], label='US_CPI_MOM_diff', marker='.', markersize=2)
plt.title('US CPI MOM Diff')
plt.xlabel('Month')
plt.ylabel('US_CPI_MOM_diff')
plt.xticks(rotation=45)
plt.grid(True)

# Plot for 'US_PERSONAL_SPENDING_PCE_MOM_diff'
plt.subplot(3, 2, 2)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['US_PERSONAL_SPENDING_PCE_diff'], label='US_PERSONAL_SPENDING_PCE_diff', marker='.', markersize=2)
plt.title('US Personal Spending PCE MOM Diff')
plt.xlabel('Month')
plt.ylabel('US_PERSONAL_SPENDING_PCE_diff')
plt.xticks(rotation=45)
plt.grid(True)

# Plot for 'US_TB_YIELD_10YRS_diff'
plt.subplot(3, 2, 3)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['US_TB_YIELD_10YRS_diff'], label='US_TB_YIELD_10YRS_diff', marker='.', markersize=2)
plt.title('US 10Y TB Yield MOM Diff')
plt.xlabel('Month')
plt.ylabel('US_TB_YIELD_10YRS_diff')
plt.xticks(rotation=45)
plt.grid(True)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()
