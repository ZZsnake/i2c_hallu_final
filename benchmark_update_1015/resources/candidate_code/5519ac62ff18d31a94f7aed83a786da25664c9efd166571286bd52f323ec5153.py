import matplotlib.pyplot as plt
import numpy as np

# Data based on the plotted population levels (population in billions)
years = np.arange(1990, 2026, 5)
africa = np.array([0.6, 0.7, 0.95, 1.1, 1.3, 1.5, 1.8, 2.1])  # Bottom layer
americas = np.array([1.2, 1.3, 1.3, 1.4, 1.4, 1.5, 1.55, 1.55])  # Second layer
asia = np.array([3.2, 3.25, 3.35, 3.6, 3.85, 3.95, 4.15, 4.5])      # Third layer
europe = np.array([0.7, 0.75, 0.75, 0.7, 0.7, 0.8, 0.8, 0.85])    # Fourth layer
oceania = np.array([0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06])   # Top layer

# Define colors (shades of red/pink, with transparency)
colors = [
    '#ffb3b3',  # Africa - lightest pink
    '#ff9999',  # Americas
    '#ff6666',  # Asia
    '#ff4d4d',  # Europe
    '#ff0000'   # Oceania - darkest red
]

# Create figure with specified dimensions
plt.figure(figsize=(10, 7))

# Plot stacked area chart
plt.stackplot(years, africa, americas, asia, europe, oceania, colors=colors, alpha=0.8)

# Labels and title
plt.xlabel("Year")
plt.ylabel("Population (Billions)")
plt.title("World Population Growth by Continent", y=1.05)

# Set axis limits
plt.xlim(1990, 2025)
plt.ylim(0, 10)

# Add legend
plt.legend(['Africa', 'Americas', 'Asia', 'Europe', 'Oceania'], loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True)

# Remove grid for clean look
plt.grid(False)

# Show plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/area_18.png", bbox_inches="tight")
