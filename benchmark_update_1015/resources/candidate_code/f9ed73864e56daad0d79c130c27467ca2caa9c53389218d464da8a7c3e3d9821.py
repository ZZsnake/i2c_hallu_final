import matplotlib.pyplot as plt
import numpy as np

# Data (approximate to match visual proportions)
categories = [
    "GDP Growth",
    "Unemployment Rate",
    "Inflation Rate",
    "NASDAQ",
    "Exchange Rate",
    "Real Estate Prices",
    "Corporate Profits"
]

regions = ["US", "EU", "China", "India", "Brazil"]

# Values chosen to visually resemble the original chart
US     = [3,  5,  2, 15,  1,  6,  8]
EU     = [5,  6,  3, 10,  4,  7,  7]
China  = [8,  7,  6, 12,  5,  8, 10]
India  = [6,  9,  4, 18,  7,  5, 12]
Brazil = [1, 10,  0,  4,  0,  1,  3]

data = np.array([US, EU, China, India, Brazil])

# Colors derived from the figure
colors = {
    "US":     "#ff6347",  # tomato-like red/orange
    "EU":     "#f5deb3",  # wheat/light tan
    "China":  "#87ceeb",  # sky blue
    "India":  "#90ee90",  # light green
    "Brazil": "#4169e1"   # royal blue
}

y_pos = np.arange(len(categories))

plt.figure(figsize=(8, 5))

# Plot stacked horizontal bars
left = np.zeros(len(categories))
for vals, region in zip(data, regions):
    plt.barh(
        y_pos,
        vals,
        left=left,
        color=colors[region],
        edgecolor='none'
    )
    left += vals

# Y-axis category labels
plt.yticks(y_pos, categories)

# Grid: vertical dashed lines similar to original
max_total = left.max()
xticks = np.arange(0, max(60, int(max_total / 10 + 2) * 10), 10)
plt.xticks(xticks)
plt.grid(axis='x', linestyle='--', linewidth=0.8, alpha=0.6)

# Legend at top center
handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[r]) for r in regions
]
plt.legend(
    handles,
    regions,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.05),
    ncol=len(regions),
    frameon=True
)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/bar_57.png", bbox_inches="tight")