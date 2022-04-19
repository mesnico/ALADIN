import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.colors import ListedColormap
import math

from matplotlib.ticker import AutoMinorLocator

x_log_scale = True

if __name__ == '__main__':
    data = {'models': ['ALADIN A/ft. + D/ft.', 'ALADIN A/ft.', 'Oscar', 'VinVL', 'TERN', 'TERAN'],
            'times': [0.023*5, 0.098*5, 2.06*5, 2.06*5, 0.019*5, 0.075*5],
            'rsums': [215.0, 224.0, 223.3, 231.4, 169.2, 204.1],
            'color_ids': [0, 1, 2, 2, 0, 1]}

    # colors_palette = ['green', 'purple', 'red']
    labels = ['disentangled (common space)', 'disentangled (alignment matrix)', 'entangled (VL Transformers)']
    colors = ListedColormap(['g', 'b', 'r'])

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    volume = 60

    # Give plot a gray background like ggplot.
    ax.set_facecolor('#EBEBEB')
    ax.set_axisbelow(True)
    # Remove border around plot.
    [ax.spines[side].set_visible(False) for side in ax.spines]
    # Style the grid.
    ax.grid(which='major', color='white', linewidth=1.2)
    ax.grid(which='minor', color='white', linewidth=0.6)
    # Show the minor ticks and grid.
    ax.minorticks_on()
    # Now hide the minor ticks (but leave the gridlines).
    ax.tick_params(which='minor', bottom=False, left=False)

    # Only show minor gridlines once in between major gridlines.
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # colors = [colors_palette[i] for i in data['color_ids']]
    # labels = [legend[i] for i in data['color_ids']]
    scatter = ax.scatter(data['times'], data['rsums'], c=data['color_ids'], cmap=colors)

    for x, y, txt, c_id in zip(data['times'], data['rsums'], data['models'], data['color_ids']):
        if x < 6:
            delta_x = 0.3
            delta_y = 0
        else:
            delta_x = -1.3 if not x_log_scale else -1.3
            delta_y = -3
        if txt == 'TERAN':
            delta_y -= 3
        coords = (x + x*(10**(delta_x/5) - 1), y + delta_y) if x_log_scale else (x + delta_x, y + delta_y)
        # ax.annotate(txt, coords, bbox=dict(boxstyle="square,pad=0.2", fc="cyan", ec="b", lw=1))
        ax.text(coords[0], coords[1], txt, style='italic',
                bbox={'facecolor': colors(c_id), 'alpha': 0.3, 'pad': 2})
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