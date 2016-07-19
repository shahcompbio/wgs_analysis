'''
# Position plots

Plotting of positions in chromosome plots using stem plots.
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn

def plot_positions(ax, positions, chrom, site):

    positions = positions[positions['chrom'] == chrom]
    positions = positions[positions['site_id'] == site]
    positions = positions[positions['alt_counts'] > 0]
    for coord in positions['coord'].unique():
        ax.plot([coord, coord], [-0.2, 0.2], color='#146414', linewidth=0.75)


def plot_breakends(ax, brks, site_id, chromosome, start=None, end=None, xlim=None, ylim=None, fcolor=None, lw=None):

    if xlim is None:
        xlim = ax.get_xlim()
    xrng = xlim[1] - xlim[0]

    if ylim is None:
        ylim = ax.get_ylim()
    yrng = ylim[1] - ylim[0]

    arrow_length = 0.01 * xrng
    
    brks = brks[(brks['chrom'] == chromosome)]

    if start is not None:
        brks = brks[(brks['coord'] > start)]
    
    if end is not None:
        brks = brks[(brks['coord'] < end)]

    brks = brks[(brks['site_id'] == site_id)]
    brks = brks[(brks['num_span'] > 0)]

    head_width = yrng / 30.
    head_length = xrng / 100.

    for strand, positions in brks.groupby('strand')['coord']:
        xpos = positions.values
        ypos = np.random.uniform(ylim[0] + 0.75*yrng,
                                 ylim[0] + 0.95*yrng,
                                 size=len(positions))
        offset = (arrow_length, -arrow_length)[strand == '+']
        for x, y in zip(xpos, ypos):
            color = 'grey'
            if fcolor is not None:
                color = fcolor(chromosome, strand, x)
            ax.arrow(x, y, offset, 0, color=color, lw=lw, alpha=1.0, head_width=head_width, head_length=head_length)
            ax.plot([x, x], [-100, 100], color=color, lw=lw, alpha=1.0)
    
