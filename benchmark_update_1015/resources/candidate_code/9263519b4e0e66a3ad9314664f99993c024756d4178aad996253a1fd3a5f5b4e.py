import matplotlib.pyplot as plt
import numpy as np

# Set figure size
plt.figure(figsize=(8, 5))

# Data for the stacked bar chart
categories = ['GDP Growth', 'Unemployment Rate', 'Inflation Rate', 'NASDAQ', 'Exchange Rate', 'Real Estate Prices', 'Corporate Profits']
US = [3, 5, 2, 15, 1, 4, 5]
China = [10, 8, 3, 20, 10, 10, 10]
Brazil = [2, 15, 5, 10, 5, 5, 5]
EU = [2, 5, 2, 10, 1, 5, 5]
India = [5, 5, 5, 20, 5, 5, 10]

# Colors for each country
colors = ['red', 'lightblue', 'blue', 'lightyellow', 'green']

# Create the stacked bar chart
x = np.arange(len(categories))  # x positions for each category
width = 0.8  # width of the bars

# Plot each country's data as a stacked bar
plt.bar(x, US, width, label='US', color=colors[0])
plt.bar(x, China, width, bottom=US, label='China', color=colors[1])
plt.bar(x, Brazil, width, bottom=[US[i] + China[i] for i in range(len(US))], label='Brazil', color=colors[2])
plt.bar(x, EU, width, bottom=[US[i] + China[i] + Brazil[i] for i in range(len(US))], label='EU', color=colors[3])
plt.bar(x, India, width, bottom=[US[i] + China[i] + Brazil[i] + EU[i] for i in range(len(US))], label='India', color=colors[4])

# Set labels and title
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Economic Indicators by Country')
plt.xticks(x, categories, rotation=45, ha='right')
plt.legend(loc='upper right')

# Set y-axis limits
plt.ylim(0, 60)

# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/bar_57.png", bbox_inches="tight")