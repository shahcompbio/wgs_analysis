import numpy as np

def plot_cn_on_ax(cn, ax, value_name="total_cn", **kwargs):
    """ Plot CN to given ax
    - cn: DataFrame with 'chromosome', 'start', 'end', and value_name (default:'total_cn')
    - ax: pyplot Axes
    - value_name: value to plot [str]
    - cn_color: color to plot value_name [str]
    - kwargs: passed to ax.plot
    """
    max_cn = cn[value_name].max()
    ax.set_ylim((-0.5, max_cn+0.5))
    ax.set_yticks(np.arange(0, max_cn+1, 1))
    ax.set_yticklabels([int(t) for t in ax.get_yticks()])
    for _, row in cn.iterrows():
        _, start, end = row['chromosome'], int(row['start']), int(row['end'])
        cn_value = float(row[value_name])
        ax.plot([start, end], [cn_value, cn_value], solid_capstyle='butt', **kwargs)