import matplotlib.pyplot as plt

# Set figure size
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Define company names and data for left plot
left_companies = ['Google', 'Samsung', 'Apple']
left_revenue_start = [70, 70, 60]
left_revenue_end = [77, 78, 70]
left_market_start = [20, 25, 35]
left_market_end = [22, 28, 40]
left_revenue_labels = [7.00, 8.00, 10.00]
left_market_labels = [2.00, 3.00, 5.00]

# Define company names and data for right plot
right_companies = ['Sony', 'Huawei', 'Microsoft']
right_revenue_start = [60, 60, 50]
right_revenue_end = [65, 66, 59]
right_market_start = [15, 20, 30]
right_market_end = [18, 25, 34]
right_revenue_labels = [5.00, 6.00, 9.00]
right_market_labels = [3.00, 5.00, 4.00]

# Common styling
colors = {'revenue': 'gray', 'market': 'purple'}
arrow_props = {'arrowstyle': '-|>', 'lw': 2}

# Plot left subplot
ax1 = axes[0]
ax1.set_yticks(range(len(left_companies)))
ax1.set_yticklabels(left_companies)
ax1.set_xlim(0, 100)
ax1.grid(True, linestyle='--', linewidth=0.5, color='gray')

for i, company in enumerate(left_companies):
    # Revenue Growth (gray arrow)
    ax1.annotate("", xy=(left_revenue_end[i], i), xytext=(left_revenue_start[i], i),
                 arrowprops={'arrowstyle': '-|>', 'color': colors['revenue'], 'lw': 2})
    ax1.plot(left_revenue_start[i], i, 'ko', markersize=4)
    ax1.plot(left_revenue_end[i], i, 'ko', markersize=4)
    ax1.text(left_revenue_end[i] + 1, i + 0.05, f"{left_revenue_labels[i]:.2f}", 
             color=colors['revenue'], fontsize=9, ha='left', va='bottom')

    # Market Share Growth (purple arrow)
    ax1.annotate("", xy=(left_market_end[i], i), xytext=(left_market_start[i], i),
                 arrowprops={'arrowstyle': '-|>', 'color': colors['market'], 'lw': 2})
    ax1.plot(left_market_start[i], i, 'ko', markersize=4)
    ax1.plot(left_market_end[i], i, 'ko', markersize=4)
    ax1.text(left_market_end[i] + 1, i + 0.05, f"{left_market_labels[i]:.2f}", 
             color=colors['market'], fontsize=9, ha='left', va='bottom')

# Plot right subplot
ax2 = axes[1]
ax2.set_yticks(range(len(right_companies)))
ax2.set_yticklabels(right_companies)
ax2.set_xlim(0, 100)
ax2.grid(True, linestyle='--', linewidth=0.5, color='gray')

for i, company in enumerate(right_companies):
    # Revenue Growth (gray arrow)
    ax2.annotate("", xy=(right_revenue_end[i], i), xytext=(right_revenue_start[i], i),
                 arrowprops={'arrowstyle': '-|>', 'color': colors['revenue'], 'lw': 2})
    ax2.plot(right_revenue_start[i], i, 'ko', markersize=4)
    ax2.plot(right_revenue_end[i], i, 'ko', markersize=4)
    ax2.text(right_revenue_end[i] + 1, i + 0.05, f"{right_revenue_labels[i]:.2f}", 
             color=colors['revenue'], fontsize=9, ha='left', va='bottom')

    # Market Share Growth (purple arrow)
    ax2.annotate("", xy=(right_market_end[i], i), xytext=(right_market_start[i], i),
                 arrowprops={'arrowstyle': '-|>', 'color': colors['market'], 'lw': 2})
    ax2.plot(right_market_start[i], i, 'ko', markersize=4)
    ax2.plot(right_market_end[i], i, 'ko', markersize=4)
    ax2.text(right_market_end[i] + 1, i + 0.05, f"{right_market_labels[i]:.2f}", 
             color=colors['market'], fontsize=9, ha='left', va='bottom')

# Add legends
legend_elements = [
    plt.Line2D([0], [0], color=colors['revenue'], lw=2, marker='>', markersize=10, label='Revenue Growth'),
    plt.Line2D([0], [0], color=colors['market'], lw=2, marker='>', markersize=10, label='Market Share Growth')
]

ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
ax2.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/quiver_7.png", bbox_inches="tight")