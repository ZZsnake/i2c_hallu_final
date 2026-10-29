import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge

fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

def draw_problem_section(label: str, x_pos: float, y_pos: float, text_x: float, text_y: float):
    ax.annotate(str.upper(label), xy=(x_pos, y_pos),
                xytext=(text_x, text_y),
                fontsize=10,
                color='white',
                weight='bold',
                xycoords='data',
                verticalalignment='center',
                horizontalalignment='center',
                textcoords='offset fontsize',
                arrowprops=dict(arrowstyle="->", facecolor='black'),
                bbox=dict(boxstyle='square',
                          facecolor='tab:red',
                          pad=0.8))

def draw_causes(cause_list: list, x_pos: float, y_pos: float, text_offset=(-9, -0.3), top: bool = True):
    for idx, cause in enumerate(cause_list):
        offset_coords = [[0.02, 0],
                         [0.23, 0.5],
                         [-0.46, -1],
                         [0.69, 1.5],
                         [-0.92, -2],
                         [1.15, 2.5]]
        
        x_pos -= offset_coords[idx][0]
        y_pos += offset_coords[idx][1] if top else -offset_coords[idx][1]

        ax.annotate(cause, xy=(x_pos, y_pos),
                    horizontalalignment='center',
                    xytext=text_offset,
                    fontsize=9,
                    xycoords='data',
                    textcoords='offset fontsize',
                    arrowprops=dict(arrowstyle="->",
                                    facecolor='black'))

def construct_diagram(data: dict):
    length = (math.ceil(len(data) / 2)) - 1
    draw_main_structure(-2 - length, 2 + length)

    position_offset = 0
    base_positions = [1.55, 0.8]
    for idx, category in enumerate(data.values()):
        is_top = idx % 2 == 0
        cause_y = 1.7 if is_top else -1.7
        text_y = 16 if is_top else -16

        problem_x = base_positions[0] + length + position_offset
        cause_x = base_positions[1] + length + position_offset
        if not is_top:
            position_offset -= 2.5
        if idx > 5:
            raise ValueError(f'Maximum number of categories is 6, provided {len(data)}')

        draw_problem_section(list(data.keys())[idx], problem_x, 0, -12, text_y)
        draw_causes(category, cause_x, cause_y, top=is_top)

def draw_main_structure(x_start: int, x_end: int):
    ax.plot([x_start - 0.1, x_end], [0, 0], color='tab:red', linewidth=3)
    ax.text(x_end + 0.1, -0.05, 'ISSUE', fontsize=10,
            weight='bold', color='white')
    semicircle = Wedge((x_end, 0), 1, 270, 90, fc='tab:red')
    ax.add_patch(semicircle)
    tail_points = [[x_start - 0.8, 0.8], [x_start - 0.8, -0.8], [x_start, -0.01]]
    triangle = Polygon(tail_points, fc='tab:red')
    ax.add_patch(triangle)

categories = {
    'Planning': ['Timeline issues', 'Budget constraints', 'Scope changes', 'Resource allocation'],
    'Design': ['Technical limitations', 'User experience', 'Aesthetics'],
    'Development': ['Coding errors', 'Integration problems', 'Testing gaps'],
    'Deployment': ['Server configuration', 'Database setup', 'Security concerns'],
    'Monitoring': ['Performance metrics', 'Error tracking', 'User feedback'],
    'Maintenance': ['Bug fixes', 'Updates', 'Technical debt']
}

construct_diagram(categories)
plt.show()