# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

# ===================
# Part 2: Data Preparation
# ===================
# Data
augmentation_levels = ["0", "0.125", "0.25", "0.5", "1", "2", "4", "8"]
processing_time = np.array([2, 1.8, 1.6, 1.5, 1.4, 1.3, 1.1, 1])
power_consumption = np.array([1, 1.5, 2, 2.2, 2.5, 2.7, 2.8, 3])
latency = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.3])

# Calculate cumulative values for stacked area chart
cumulative_processing_time = processing_time
cumulative_power_consumption = cumulative_processing_time + power_consumption
cumulative_latency = cumulative_power_consumption + latency

# Positions for the bars on the x-axis
ind = np.arange(len(augmentation_levels))

# Variables for plot configuration
processing_time_label = "Processing Time"
power_consumption_label = "Power Consumption"
latency_label = "Latency"
xlabel_text = "Augmentation Level"
ylabel_text = "Performance Metrics"
title_text = "Cumulative Performance Metrics by Augmentation Level"


# ===================
# Part 3: Plot Configuration and Rendering
# ===================
xlim_values = (0, 7)
ylim_values = (0, 10)
yticks_values = np.arange(0, 11, 1)
legend_location = "upper left"
legend_fontsize = 12
legend_frameon = False
legend_shadow = True
legend_facecolor = "#ffffff"
legend_ncol = 1
legend_bbox_to_anchor = (1.05, 1)
# Plot
figure, axes = plt.subplots(figsize=(10, 6))  # Adjusted for better aspect ratio

axes.fill_between(
    augmentation_levels, 0, cumulative_processing_time, label=processing_time_label, color="#d73027", alpha=0.7
)
axes.fill_between(
    augmentation_levels,
    cumulative_processing_time,
    cumulative_power_consumption,
    label=power_consumption_label,
    color="#fc8d59",
    alpha=0.7,
)
axes.fill_between(
    augmentation_levels,
    cumulative_power_consumption,
    cumulative_latency,
    label=latency_label,
    color="#4575b4",
    alpha=0.7,
)

# Enhancing the plot with additional visuals
axes.spines["top"].set_visible(False)
axes.spines["right"].set_visible(False)
axes.spines["left"].set_visible(False)
axes.spines["bottom"].set_visible(False)

# Setting the x-axis and y-axis limits dynamically
axes.set_ylim(*ylim_values)  # Ensure all data fits well
axes.set_xlim(*xlim_values)

# Labels, Title and Grid
axes.set_xlabel(xlabel_text, fontsize=14)
axes.set_ylabel(ylabel_text, fontsize=14)
axes.set_title(title_text, fontsize=16, y=1.05)
axes.tick_params(axis="both", which="both", color="gray")

# Custom legend
axes.legend(
    loc=legend_location,
    fontsize=legend_fontsize,
    frameon=legend_frameon,
    shadow=legend_shadow,
    facecolor=legend_facecolor,
    ncol=legend_ncol,
    bbox_to_anchor=legend_bbox_to_anchor,
)

# Grid
axes.grid(True, linestyle="--", alpha=0.5, which="both")

# ===================
# Part 4: Saving Output
# ===================
# Adjusting layout to reduce white space
plt.tight_layout()
plt.savefig("area_14.pdf", bbox_inches="tight")
