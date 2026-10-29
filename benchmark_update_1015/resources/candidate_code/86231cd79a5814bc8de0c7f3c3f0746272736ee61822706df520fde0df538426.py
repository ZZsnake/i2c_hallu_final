import matplotlib.pyplot as plt
import squarify

# Data
brands = ['Cargill', 'Wilmar International', 'ADM', 'Bunge', 'COFCO', 'Others']
market_share = [30, 25, 20, 10, 8, 7]

# Colors matching the image
colors = ['#2E8B57', '#FF6347', '#4682B4', '#DAA520', '#8A2BE2', '#FF69B4']

# Create figure with specified dimensions
fig, ax = plt.subplots(figsize=(10, 8))

# Create treemap
squarify.plot(sizes=market_share, 
              label=[f'{brand}\n{share}%' for brand, share in zip(brands, market_share)],
              color=colors,
              alpha=0.8,
              text_kwargs={'fontsize': 14, 'weight': 'bold', 'color': 'white'},
              edgecolor='black',
              linewidth=1,
              pad=True,
              ax=ax)

# Set title
ax.set_title('Market Share of Edible Oil Brands', fontsize=16, weight='bold', pad=6)

# Remove axes
ax.axis('off')

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/tree_20.png", bbox_inches="tight")
