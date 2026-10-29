# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt

# ===================
# Part 2: Data Preparation
# ===================
# Sample data for the boxplots
t2tt_bleu_values = [19, 20, 21, 22, 21.5, 20.5, 19.5]
s2tt_bleu_values = [18, 19, 20, 19.5, 18.5, 19.2, 18.8]
x_tick_labels = ["Llama-70b\n(T2TT)", "Llama-70b\n(S2TT)"]
y_axis_label = "BLEU"
x_axis_limits = [0.5, 3]
y_axis_limits = [16, 23]
y_axis_ticks = [16, 18, 20, 22]
annotation_text = "1.0 BLEU "
horizontal_line_values = [20.5, 19.5]


# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Create a figure and axis with the specified dimensions
fig, ax = plt.subplots(figsize=(6, 4))

# Create the boxplots
boxplot_t2tt = ax.boxplot(
    t2tt_bleu_values, positions=[1], widths=0.6, patch_artist=True, boxprops=dict(facecolor="none")
)
boxplot_s2tt = ax.boxplot(
    s2tt_bleu_values, positions=[2], widths=0.6, patch_artist=True, boxprops=dict(facecolor="none")
)

# Set the labels for the x-axis
ax.set_xticklabels(x_tick_labels, ha="center")
ax.set_xlim()

# Set the labels for the y-axis
ax.set_ylabel(y_axis_label)

# Set the y-axis limits
ax.set_ylim(y_axis_limits)
ax.set_yticks(y_axis_ticks)

# Add a custom annotation for the 1.0 BLEU difference
ax.annotate(
    "",
    xy=(2.25, 20.5),
    xytext=(2.25, 19.0),
    arrowprops=dict(facecolor="black", arrowstyle="->"),
    horizontalalignment="center",
)

# Add the text for the 1.0 BLEU difference
ax.text(2.5, 19.7, annotation_text, horizontalalignment="center")

# Add the dashed line for the 1.0 BLEU difference
ax.hlines(horizontal_line_values[0], 1, 2.25, colors="grey", linestyles="dashed")
ax.hlines(horizontal_line_values[1], 2, 2.25, colors="grey", linestyles="dashed")

# ===================
# Part 4: Saving Output
# ===================
# Adjust the layout and show the plot
plt.tight_layout()
plt.savefig("box_11.pdf", bbox_inches="tight")
