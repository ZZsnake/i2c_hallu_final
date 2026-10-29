import matplotlib.pyplot as plt
import numpy as np

# Define data
indicators = ['GDP Growth', 'Unemployment Rate', 'Inflation Rate', 
              'NASDAQ', 'Exchange Rate', 'Real Estate Prices', 'Corporate Profits']
us_data = [8, 6, 2, 15, 1, 7, 9]
china_data = [12, 10, 3, 20, 10, 15, 18]
brazil_data = [10, 15, 5, 12, 4, 2, 6]
eu_data = [5, 8, 1, 18, 2, 4, 10]
india_data = [15, 12, 7, 25, 8, 20, 22]

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Plotting data
bar_width = 0.15
x = np.arange(len(indicators))
ax.bar(x, us_data, width=bar_width, label='US', color='red')
ax.bar(x + bar_width, china_data, width=bar_width, label='China', color='blue')
ax.bar(x + 2*bar_width, brazil_data, width=bar_width, label='Brazil', color='lightblue')
ax.bar(x + 3*bar_width, eu_data, width=bar_width, label='EU', color='orange')
ax.bar(x + 4*bar_width, india_data, width=bar_width, label='India', color='green')

# Adding text on top of bars
for i, v in enumerate(us_data):
    ax.text(i * 5 + bar_width / 2, v + 0.5, str(v), ha='center', va='bottom')
for i, v in enumerate(china_data):
    ax.text((i + 1) * 5 + bar_width / 2, v + 0.5, str(v), ha='center', va='bottom')
for i, v in enumerate(brazil_data):
    ax.text((i + 2) * 5 + bar_width / 2, v + 0.5, str(v), ha='center', va='bottom')
for i, v in enumerate(eu_data):
    ax.text((i + 3) * 5 + bar_width / 2, v + 0.5, str(v), ha='center', va='bottom')
for i, v in enumerate(india_data):
    ax.text((i + 4) * 5 + bar_width / 2, v + 0.5, str(v), ha='center', va='bottom')

# Setting labels and title
ax.set_xlabel('Value')
ax.set_ylabel('Economic Indicators')
ax.set_title('Comparison of Economic Indicators Across Regions')
ax.set_xticks(x + 2.5 * bar_width)
ax.set_xticklabels(indicators)

# Show legend
plt.legend()

# Display plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/bar_57.png", bbox_inches="tight")