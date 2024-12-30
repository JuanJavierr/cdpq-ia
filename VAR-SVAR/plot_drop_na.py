# plot_data.py
import matplotlib.pyplot as plt
from load_ import load_and_process_data  # Import the function from load_data.py

# File path to the data (update the path to your actual file)
file_path = 'Book1.xlsx'

# Load the data
monthly_avg = load_and_process_data(file_path)

monthly_avg['US_CPI_diff'] = monthly_avg['US_CPI'].diff().dropna()
monthly_avg['US_PERSONAL_SPENDING_PCE_diff'] = monthly_avg['US_PERSONAL_SPENDING_PCE'].diff().dropna()
monthly_avg['SNP_500_diff'] = monthly_avg['SNP_500'].diff().dropna()
monthly_avg['FFED_diff'] = monthly_avg['FFED'].diff().dropna()
monthly_avg['US_TB_YIELD_10YRS_diff'] = monthly_avg['US_TB_YIELD_10YRS'].diff().dropna()
monthly_avg['US_TB_YIELD_1YR_diff'] = monthly_avg['US_TB_YIELD_1YR'].diff().dropna()




plt.figure(figsize=(12, 12))


# Plot for 'US_CPI_diff'
plt.subplot(4, 2, 1)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['US_CPI_diff'], label='US_CPI_diff', marker='.', markersize=2)
plt.title('US CPI (Differenced)')
plt.xlabel('Month')
plt.ylabel('US_CPI_diff')
plt.xticks(rotation=45)
plt.grid(True)

# Plot for 'US_PERSONAL_SPENDING_PCE_diff'
plt.subplot(4, 2, 2)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['US_PERSONAL_SPENDING_PCE_diff'], label='US_PERSONAL_SPENDING_PCE_diff', marker='.', markersize=2)
plt.title('US Personal Spending PCE (Differenced)')
plt.xlabel('Month')
plt.ylabel('US_PERSONAL_SPENDING_PCE_diff')
plt.xticks(rotation=45)
plt.grid(True)

# Plot for 'SNP_500_diff'
plt.subplot(4, 2, 3)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['SNP_500_diff'], label='SNP_500_diff', marker='.', markersize=2)
plt.title('S&P 500 (Differenced)')
plt.xlabel('Month')
plt.ylabel('SNP_500_diff')
plt.xticks(rotation=45)
plt.grid(True)

# Plot for 'FFED_diff'
plt.subplot(4, 2, 4)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['FFED_diff'], label='FFED_diff', marker='.', markersize=2)
plt.title('Fed Funds Rate (Differenced)')
plt.xlabel('Month')
plt.ylabel('FFED_diff')
plt.xticks(rotation=45)
plt.grid(True)

# Plot for 'US_TB_YIELD_10YRS_diff'
plt.subplot(4, 2, 5)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['US_TB_YIELD_10YRS_diff'], label='US_TB_YIELD_10YRS_diff', marker='.', markersize=2)
plt.title('US 10-Year TB Yield (Differenced)')
plt.xlabel('Month')
plt.ylabel('US_TB_YIELD_10YRS_diff')
plt.xticks(rotation=45)
plt.grid(True)

# Plot for 'US_TB_YIELD_1YRS_diff'
plt.subplot(4, 2, 6)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['US_TB_YIELD_1YR_diff'], label='US_TB_YIELD_1YRS_diff', marker='.', markersize=2)
plt.title('US 1-Year TB Yield (Differenced)')
plt.xlabel('Month')
plt.ylabel('US_TB_YIELD_1YRS_diff')
plt.xticks(rotation=45)
plt.grid(True)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()
