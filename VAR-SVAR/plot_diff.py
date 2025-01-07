# plot_data.py
import matplotlib.pyplot as plt
from load_ import load_and_process_data  # Import the function from load_data.py

# File path to the data (update the path to your actual file)
file_path = 'Book1.xlsx'

# Load the data
monthly_avg = load_and_process_data(file_path)

# Recompute differenced columns correctly
monthly_avg['US_CPI_diff'] = monthly_avg['US_CPI'].diff()
monthly_avg['US_PERSONAL_SPENDING_PCE_diff'] = monthly_avg['US_PERSONAL_SPENDING_PCE'].diff()
monthly_avg['SNP_500_diff'] = monthly_avg['SNP_500'].diff()
monthly_avg['FFED_diff'] = monthly_avg['FFED'].diff()
monthly_avg['US_TB_YIELD_10YRS_diff'] = monthly_avg['US_TB_YIELD_10YRS'].diff()
monthly_avg['US_TB_YIELD_1YRS_diff'] = monthly_avg['US_TB_YIELD_1YR'].diff()

# Drop NaN values to align index
monthly_avg_diff = monthly_avg.dropna()




plt.figure(figsize=(1, 1))

# Plot for 'US_TB_YIELD_10YRS_diff'
plt.subplot(1, 1, 1)
plt.plot(monthly_avg_diff.index.to_timestamp(), monthly_avg_diff['US_TB_YIELD_10YRS_diff'], label='US_TB_YIELD_10YRS_diff', marker='.', markersize=2)
plt.title('US 10-Year TB Yield (Differenced)')
plt.xlabel('Month')
plt.ylabel('US_TB_YIELD_10YRS_diff')
plt.xticks(rotation=45)
plt.grid(True)



# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()
