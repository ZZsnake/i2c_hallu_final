# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ===================
# Part 2: Data Preparation
# ===================
# Data for the plot (Technology Companies)
companies_1 = ["Apple", "Samsung", "Google"]
market_share_initial_1 = [35, 25, 20]
market_share_change_1 = [5, 3, 2]
revenue_initial_1 = [60, 80, 70]
revenue_change_1 = [10, 8, 7]
ax1_labels = ["Revenue Growth", "Market Share Growth"]

companies_2 = ["Microsoft", "Huawei", "Sony"]
market_share_initial_2 = [30, 20, 15]
market_share_change_2 = [4, 5, 3]
revenue_initial_2 = [50, 65, 60]
revenue_change_2 = [9, 6, 5]
ax2_labels = ["Revenue Growth", "Market Share Growth"]
# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Set the y-axis offsets to be in the middle of each grid
offset = 0.5

# First subplot (companies_1)
for i, company in enumerate(companies_1):
    # Revenue growth line with arrow and black dots at start and end
    ax1.annotate(
        "",
        xy=(revenue_initial_1[i], i + offset * 3 / 2),
        xytext=(revenue_initial_1[i] + revenue_change_1[i], i + offset * 3 / 2),
        arrowprops=dict(arrowstyle="<-", color="gray"),
    )
    ax1.scatter(
        [revenue_initial_1[i], revenue_initial_1[i] + revenue_change_1[i]],
        [i + offset * 3 / 2, i + offset * 3 / 2],
        color="black",
        s=10,
    )
    ax1.annotate(
        f"{revenue_change_1[i]:.2f}",
        (revenue_initial_1[i] + revenue_change_1[i], i + offset * 1.75),
        color="gray",
        ha="right",
        va="center",
    )

    # Market share growth line with arrow and black dots at start and end
    ax1.annotate(
        "",
        xy=(market_share_initial_1[i], i + offset / 2),
        xytext=(market_share_initial_1[i] + market_share_change_1[i], i + offset / 2),
        arrowprops=dict(arrowstyle="<-", color="darkorchid"),
    )
    ax1.scatter(
        [market_share_initial_1[i], market_share_initial_1[i] + market_share_change_1[i]],
        [i + offset / 2, i + offset / 2],
        color="black",
        s=10,
    )
    ax1.annotate(
        f"{market_share_change_1[i]:.2f}",
        (market_share_initial_1[i] + market_share_change_1[i], i + offset * 0.75),
        color="darkorchid",
        ha="left",
        va="center",
    )

# Second subplot (companies_2)
for i, company in enumerate(companies_2):
    ax2.annotate(
        "",
        xy=(revenue_initial_2[i], i + offset * 3 / 2),
        xytext=(revenue_initial_2[i] + revenue_change_2[i], i + offset * 3 / 2),
        arrowprops=dict(arrowstyle="<-", color="gray"),
    )
    ax2.scatter(
        [revenue_initial_2[i], revenue_initial_2[i] + revenue_change_2[i]],
        [i + offset * 3 / 2, i + offset * 3 / 2],
        color="black",
        s=10,
    )
    ax2.annotate(
        f"{revenue_change_2[i]:.2f}",
        (revenue_initial_2[i] + revenue_change_2[i], i + offset * 1.75),
        color="gray",
        ha="right",
        va="center",
    )

    ax2.annotate(
        "",
        xy=(market_share_initial_2[i], i + offset / 2),
        xytext=(market_share_initial_2[i] + market_share_change_2[i], i + offset / 2),
        arrowprops=dict(arrowstyle="<-", color="darkorchid"),
    )
    ax2.scatter(
        [market_share_initial_2[i], market_share_initial_2[i] + market_share_change_2[i]],
        [i + offset / 2, i + offset / 2],
        color="black",
        s=10,
    )
    ax2.annotate(
        f"{market_share_change_2[i]:.2f}",
        (market_share_initial_2[i] + market_share_change_2[i], i + offset * 0.75),
        color="darkorchid",
        ha="left",
        va="center",
    )

# Set y-axis limits
ax1.set_ylim(0, len(companies_1))
ax2.set_ylim(0, len(companies_2))

# Set x-axis limits uniformly
ax1.set_xlim(0, 100)
ax2.set_xlim(0, 100)

# Adjust the y-axis tick positions
ax1.set_yticks([i + offset for i in range(len(companies_1))])
ax1.set_yticklabels(companies_1)
ax2.set_yticks([i + offset for i in range(len(companies_2))])
ax2.set_yticklabels(companies_2)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")

# Offset grid lines on the y-axis
ax1.set_yticks([i for i in range(len(companies_1))], minor=True)
ax2.set_yticks([i for i in range(len(companies_2))], minor=True)
ax1.yaxis.grid(True, which="minor", linewidth=0.5, alpha=0.7, color="black")
ax2.yaxis.grid(True, which="minor", linewidth=0.5, alpha=0.7, color="black")

# Add x-axis grid lines and set gap to 10
ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
ax2.xaxis.set_major_locator(plt.MultipleLocator(10))
ax1.grid(axis="x", linestyle="--", linewidth=0.5)
ax2.grid(axis="x", linestyle="--", linewidth=0.5)

# Create arrow-shaped legend entries with a line that aligns with the arrowhead
gray_arrow = mlines.Line2D(
    [],
    [],
    color="gray",
    marker=">",
    linestyle="-",
    markersize=8,
    label=ax1_labels[0],
    linewidth=2,
    markeredgewidth=2,
    markevery=(1, 1),
)
darkorchid_arrow = mlines.Line2D(
    [],
    [],
    color="darkorchid",
    marker=">",
    linestyle="-",
    markersize=8,
    label=ax1_labels[1],
    linewidth=2,
    markeredgewidth=2,
    markevery=(1, 1),
)
fig.legend(handles=[gray_arrow, darkorchid_arrow], bbox_to_anchor=(0.45, 0), ncol=2)

gray_arrow = mlines.Line2D(
    [],
    [],
    color="gray",
    marker=">",
    linestyle="-",
    markersize=8,
    label=ax2_labels[0],
    linewidth=2,
    markeredgewidth=2,
    markevery=(1, 1),
)
darkorchid_arrow = mlines.Line2D(
    [],
    [],
    color="darkorchid",
    marker=">",
    linestyle="-",
    markersize=8,
    label=ax2_labels[1],
    linewidth=2,
    markeredgewidth=2,
    markevery=(1, 1),
)
fig.legend(handles=[gray_arrow, darkorchid_arrow], bbox_to_anchor=(0.85, 0), ncol=2)

# ===================
# Part 4: Saving Output
# ===================
# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('quiver_7.pdf', bbox_inches='tight')