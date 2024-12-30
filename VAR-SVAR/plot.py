# plot_data.py
import matplotlib.pyplot as plt
from load_ import load_and_process_data  # Import the function from load_data.py

# File path to the data (update the path to your actual file)
file_path = 'Book1.xlsx'

# Load the data
monthly_avg = load_and_process_data(file_path)

# Create individual plots
plt.figure(figsize=(12, 10))

# Plot for 'US_CPI_MOM'
plt.subplot(4, 2, 1)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['US_CPI'], label='US_CPI', marker='.', markersize=2)
plt.title('US CPI')
plt.xlabel('Month')
plt.ylabel('US_CPI_MOM')
plt.xticks(rotation=45)
plt.grid(True)

# Plot for 'US_PERSONAL_SPENDING_PCE_MOM'
plt.subplot(4, 2, 2)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['US_PERSONAL_SPENDING_PCE'], label='US_PERSONAL_SPENDING_PCE', marker='.', markersize=2)
plt.title('US Personal Spending PCE')
plt.xlabel('Month')
plt.ylabel('US_PERSONAL_SPENDING_PCE')
plt.xticks(rotation=45)
plt.grid(True)

# Plot for 'US_UNEMPLOYMENT_RATE_MOM'
plt.subplot(4, 2, 3)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['US_UNEMPLOYMENT_RATE'], label='US_UNEMPLOYMENT_RATE', marker='.', markersize=3)
plt.title('US Unemployment Rate')
plt.xlabel('Month')
plt.ylabel('US_UNEMPLOYMENT_RATE')
plt.xticks(rotation=45)
plt.grid(True)

# Plot for 'SNP_500_MOM'
plt.subplot(4, 2, 4)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['SNP_500'], label='SNP_500', marker='.', markersize=3)
plt.title('SNP 500')
plt.xlabel('Month')
plt.ylabel('SNP_500')
plt.xticks(rotation=45)
plt.grid(True)

# Plot for 'FFED_MOM'
plt.subplot(4, 2, 5)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['FFED'], label='FFED', marker='.', markersize=3)
plt.title('FFED')
plt.xlabel('Month')
plt.ylabel('FFED')
plt.xticks(rotation=45)
plt.grid(True)

# Plot for 'US_TB_YIELD_10YRS'
plt.subplot(4, 2, 6)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['US_TB_YIELD_10YRS'], label='US_TB_YIELD_10YRS', marker='.', markersize=3)
plt.title('US 10Y TB Yield')
plt.xlabel('Month')
plt.ylabel('US_TB_YIELD_10YRS')
plt.xticks(rotation=45)
plt.grid(True)

# Plot for 'US_TB_YIELD_10YRS'
plt.subplot(4, 2, 7)
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['US_TB_YIELD_1YR'], label='US_TB_YIELD_1YR', marker='.', markersize=3)
plt.title('US TB YIELD 1YR')
plt.xlabel('Month')
plt.ylabel('US_TB_YIELD_1YR')
plt.xticks(rotation=45)
plt.grid(True)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()
