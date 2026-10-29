import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

def g(t, magnitude, periodicity):
    return magnitude * np.cos(2 * np.pi * periodicity * t)

t = np.linspace(0, 2, 1200)

init_magnitude = 4
init_periodicity = 2.5

fig, ax = plt.subplots()
curve, = ax.plot(t, g(t, init_magnitude, init_periodicity), 
                 color='#2E86AB', linewidth=3, linestyle='--')
ax.set_xlabel('Time Interval (s)', fontsize=12, fontfamily='serif')
ax.set_ylabel('Displacement (m)', fontsize=12, fontfamily='serif')
ax.set_title('Oscillatory Motion Analysis', fontsize=14, fontweight='bold')
ax.grid(True, linestyle=':', alpha=0.7)

fig.subplots_adjust(left=0.3, bottom=0.25)

ax_period = fig.add_axes([0.3, 0.1, 0.6, 0.03])
period_slider = Slider(
    ax=ax_period,
    label='Periodicity (Hz)',
    valmin=0.5,
    valmax=25,
    valinit=init_periodicity,
    track_color='#A23B72'
)

ax_magn = fig.add_axes([0.1, 0.25, 0.025, 0.65])
magn_slider = Slider(
    ax=ax_magn,
    label="Magnitude",
    valmin=0,
    valmax=8,
    valinit=init_magnitude,
    orientation="vertical",
    track_color='#F18F01'
)

def refresh(val):
    curve.set_ydata(g(t, magn_slider.val, period_slider.val))
    fig.canvas.draw_idle()

period_slider.on_changed(refresh)
magn_slider.on_changed(refresh)

reset_ax = fig.add_axes([0.82, 0.03, 0.12, 0.045])
reset_btn = Button(reset_ax, 'Initialize', hovercolor='#C73E1D')

def initialize(event):
    period_slider.reset()
    magn_slider.reset()

reset_btn.on_clicked(initialize)
plt.show()