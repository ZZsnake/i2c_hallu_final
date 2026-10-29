import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Create figure and main plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.15, bottom=0.3, right=0.9, top=0.9)

# Initial parameters
magnitude = 4.0
periodicity = 2.5  # Hz

# Time data
t = np.linspace(0, 2.0, 1000)
y = magnitude * np.sin(2 * np.pi * periodicity * t)

# Plot the oscillatory motion
line, = ax.plot(t, y, '--', color='steelblue', linewidth=2.5)
ax.set_xlim(0, 2.0)
ax.set_ylim(-4, 4)
ax.set_xlabel('Time Interval (s)')
ax.set_ylabel('Displacement (m)')
ax.set_title('Oscillatory Motion Analysis', fontsize=16, fontweight='bold')
ax.grid(True, linestyle=':', color='lightgray')

# Add vertical magnitude slider
ax_mag = plt.axes([0.1, 0.9, 0.03, 0.2], facecolor='lightgray')
mag_slider = Slider(
    ax=ax_mag,
    label='Magnitude',
    valmin=0.1,
    valmax=10.0,
    valinit=magnitude,
    orientation='vertical',
    color='white',
    valfmt='%1.1f'
)

# Create a gradient for the magnitude slider
mag_slider.ax.set_facecolor('none')
mag_slider.poly.set_facecolor('none')
mag_slider.ax.set_ylim(0, 10)
mag_slider.ax.set_yticks([])
mag_slider.ax.set_yticklabels([])
mag_slider.ax.spines['left'].set_visible(False)
mag_slider.ax.spines['right'].set_visible(False)
mag_slider.ax.spines['top'].set_visible(False)
mag_slider.ax.spines['bottom'].set_visible(False)

# Add horizontal periodicity slider
ax_per = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgray')
per_slider = Slider(
    ax=ax_per,
    label='Periodicity (Hz)',
    valmin=0.1,
    valmax=10.0,
    valinit=periodicity,
    color='white',
    valfmt='%1.1f'
)

# Create a gradient for the periodicity slider
per_slider.ax.set_facecolor('none')
per_slider.poly.set_facecolor('none')
per_slider.ax.set_xticks([])
per_slider.ax.set_xticklabels([])
per_slider.ax.spines['left'].set_visible(False)
per_slider.ax.spines['right'].set_visible(False)
per_slider.ax.spines['top'].set_visible(False)
per_slider.ax.spines['bottom'].set_visible(False)

# Add initialize button
ax_init = plt.axes([0.75, 0.05, 0.1, 0.04])
init_button = Button(ax_init, 'Initialize', color='lightgray', hovercolor='lightblue')

# Update function for sliders
def update(val):
    mag = mag_slider.val
    per = per_slider.val
    y = mag * np.sin(2 * np.pi * per * t)
    line.set_ydata(y)
    ax.set_ylim(-mag - 0.5, mag + 0.5)
    fig.canvas.draw_idle()

# Connect sliders to update function
mag_slider.on_changed(update)
per_slider.on_changed(update)

# Initialize button function
def init_button_clicked(event):
    mag_slider.set_val(4.0)
    per_slider.set_val(2.5)
    update(4.0)

init_button.on_clicked(init_button_clicked)

# Add gradient to magnitude slider
def draw_mag_gradient():
    ax_mag = mag_slider.ax
    ax_mag.set_facecolor('none')
    for i in range(10):
        y = i * 1.0
        ax_mag.add_patch(plt.Rectangle((0, y), 0.03, 1.0, color=plt.cm.viridis(y / 10.0), alpha=0.8))

draw_mag_gradient()

# Add gradient to periodicity slider
def draw_per_gradient():
    ax_per = per_slider.ax
    ax_per.set_facecolor('none')
    for i in range(10):
        x = i * 1.0
        ax_per.add_patch(plt.Rectangle((x, 0), 1.0, 0.03, color=plt.cm.plasma(i / 10.0), alpha=0.8))

draw_per_gradient()

# Show the plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/3e2ce01d8c774a18b12812bfc916d361.png", bbox_inches="tight")