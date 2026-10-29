import matplotlib.pyplot as plt

# Data
brands = ['Wilmar International', 'Cargill', 'ADM', 'Bunge', 'COFCO', 'Others']
market_shares = [25, 30, 20, 10, 8, 7]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A825', '#967ADC', '#FF80AB']

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Plotting the bars
bars = ax.bar(brands, market_shares, color=colors)

# Adding text on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height,
            f'{height}%', ha='center', va='bottom')

# Title and labels
ax.set_title('Market Share of Edible Oil Brands', fontsize=16)
ax.set_ylabel('Market Share (%)', fontsize=12)

# Remove x-axis ticks
ax.set_xticks([])

# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/tree_20.png", bbox_inches="tight")