import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.colors import ListedColormap
import math

x_log_scale = False

if __name__ == '__main__':
    data = {'models': ['AlaD-matching (ours)', 'AlaD-alignment (ours)', 'Oscar', 'VinVL', 'TERN', 'TERAN'],
            'times': [0.023*5, 0.098*5, 2.06*5, 2.06*5, 0.019*5, 0.075*5],
            'rsums': [209.0, 224.0, 223.3, 231.4, 169.2, 204.1],
            'color_ids': [0, 1, 2, 2, 0, 1]}

    # colors_palette = ['green', 'purple', 'red']
    labels = ['disentangled (common space)', 'disentangled (alignment matrix)', 'entangled (VL Transformers)']
    colors = ListedColormap(['g', 'b', 'r'])

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    volume = 60
    # close = 'purple'
    ax.grid(True)
    # colors = [colors_palette[i] for i in data['color_ids']]
    # labels = [legend[i] for i in data['color_ids']]
    scatter = ax.scatter(data['times'], data['rsums'], c=data['color_ids'], cmap=colors)

    for x, y, txt in zip(data['times'], data['rsums'], data['models']):
        if x < 6:
            delta_x = 0.3
            delta_y = 0
        else:
            delta_x = -1.3
            delta_y = -3
        if txt == 'TERAN':
            delta_y -= 5
        coords = (10**(x + delta_x - 1), y + delta_y) if x_log_scale else (x + delta_x, y + delta_y)
        ax.annotate(txt, coords)

    if x_log_scale:
        ax.set_xscale('log')
    ax.set_xlabel('System Latency (s)')
    ax.set_ylabel('rsum')
    handles, _ = scatter.legend_elements()
    ax.legend(handles, labels, loc="lower right", title='Method')
    # ax.set_title('Volume and percent change')

    fig.tight_layout()

    plt.savefig('efficiency-vs-effectiveness.pdf')
    plt.show()