import matplotlib.pyplot as plt

# Data
companies = ['BYD', 'Tesla', 'Volkswagen', 'Ford', 'NIO', 'Others']
market_shares = [25, 35, 15, 10, 10, 5]
colors = ['#2c3e50', '#2980b9', '#7f8c8d', '#e74c3c', '#a23cd5', '#e67e22']

# Create a figure with specified dimensions
fig, ax = plt.subplots(figsize=(14, 10))

# Plotting the bars
bars = ax.bar(companies, market_shares, color=colors)

# Adding text on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., yval + 1, f'{yval}%', ha='center', va='bottom')

# Set title and labels
ax.set_title('Car Market Share', fontsize=16)
ax.set_ylabel('Market Share (%)')
ax.set_xlabel('Companies')

# Remove ticks from x-axis
ax.set_xticks([])
ax.set_yticks([])

# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/tree_8.png", bbox_inches="tight")