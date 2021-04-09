import seaborn as sb
import numpy as np


def setup_plot():
    """
    Set up matplotlib in a consistent way between plots.
    """
    sb.set_style('ticks', {'font.sans-serif':['Helvetica']})


def setup_axes(ax):
    """
    Set up axes in a consistent way between plots.  Spines are at left and bottom, offset by 10pts.
    Other spines not visible.  Ticks only for left and bottom.
    """
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    
    ax.xaxis.grid(True, which="major", linestyle=':')
    ax.yaxis.grid(True, which="major", linestyle=':')

    return ax


def shift_lims(ax, x, y):
    """
    Shift the x and y limits in pixel space to make room for plot elements on the boundary of the plot.
    """
    x_dev, y_dev = ax.transData.inverted().transform_point((x,y)) - ax.transData.inverted().transform_point((0,0))

    ax.set_xlim([ax.get_xlim()[0] - x_dev, ax.get_xlim()[1] + x_dev])
    ax.set_ylim([ax.get_ylim()[0] - y_dev, ax.get_ylim()[1] + y_dev])


def _filter_ticks(ticks, data_min, data_max):
    ticks_before = list(filter(lambda a: a < data_min, ticks))
    ticks_within = list(filter(lambda a: a >= data_min and a <= data_max, ticks))
    ticks_after = list(filter(lambda a: a > data_max, ticks))

    ticks = []
    if len(ticks_before) > 0:
        ticks.append(max(ticks_before))
    ticks.extend(ticks_within)
    if len(ticks_after) > 0:
        ticks.append(min(ticks_after))

    return ticks


def set_xlim_filter_ticks(ax, xmin, xmax):
    """
    Set x limits and remove ticks outside of limits.
    """
    ax.set_xlim((xmin, xmax))
    ax.set_xticks(_filter_ticks(ax.get_xticks(), xmin, xmax))


def set_ylim_filter_ticks(ax, ymin, ymax):
    """
    Set y limits and remove ticks outside of limits.
    """
    ax.set_ylim((ymin, ymax))
    ax.set_yticks(_filter_ticks(ax.get_yticks(), ymin, ymax))


def trim_spines_to_ticks(ax):
    """
    Trim spines to start and end at first and last ticks.
    """
    xticks = np.concatenate((ax.get_xticks(), ax.get_xticks(minor=True)))
    ax.spines['bottom'].set_bounds(min(xticks), max(xticks))
    yticks = np.concatenate((ax.get_yticks(), ax.get_yticks(minor=True)))
    ax.spines['left'].set_bounds(min(yticks), max(yticks))


