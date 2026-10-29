import matplotlib.pyplot as plt
import numpy as np

# Create data
years = np.arange(1990, 2026, 1)

# Population data in billions (estimated based on visual inspection)
control_years = np.array([1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025])
africa = np.interp(years, control_years, [0.63, 0.72, 0.92, 1.10, 1.28, 1.48, 1.80, 2.10])
americas = np.interp(years, control_years, [1.17, 1.26, 1.33, 1.37, 1.42, 1.47, 1.52, 1.55])
asia = np.interp(years, control_years, [3.20, 3.27, 3.35, 3.58, 3.75, 3.95, 4.18, 4.50])
europe = np.interp(years, control_years, [0.72, 0.73, 0.78, 0.77, 0.75, 0.76, 0.75, 0.80])
oceania = np.interp(years, control_years, [0.03, 0.02, 0.02, 0.03, 0.05, 0.04, 0.05, 0.05])

# Create figure with specified dimensions
fig, ax = plt.subplots(figsize=(10, 7))

# Create stacked area chart
ax.fill_between(years, 0, africa, 
                color='#ffb3b3', label='Africa', linewidth=0, alpha=0.8)
ax.fill_between(years, africa, africa + americas, 
                color='#ff9999', label='Americas', linewidth=0, alpha=0.8)
ax.fill_between(years, africa + americas, africa + americas + asia, 
                color='#ff6666', label='Asia', linewidth=0, alpha=0.8)
ax.fill_between(years, africa + americas + asia, 
                africa + americas + asia + europe, 
                color='#ff4d4d', label='Europe', linewidth=0, alpha=0.8)
ax.fill_between(years, africa + americas + asia + europe, 
                africa + americas + asia + europe + oceania, 
                color='#ff0000', label='Oceania', linewidth=0, alpha=0.8)

# Set labels and title
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Population (Billions)', fontsize=11)
ax.set_title('World Population Growth by Continent', fontsize=14, pad=20)

# Set axis limits
ax.set_xlim(1990, 2025)
ax.set_ylim(0, 10)

# Add legend
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)

# Keep full rectangular axes frame
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

# Adjust layout to prevent legend cutoff
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/area_18.png", bbox_inches="tight")
