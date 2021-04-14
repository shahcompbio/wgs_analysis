'''
# Rearrangement plotting

'''

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import random
import scipy

import wgs_analysis.plots as plots
import wgs_analysis.refgenome as refgenome
import wgs_analysis.plots.colors


default_rearrangement_types = ['foldback', 'deletion', 'duplication', 'inversion', 'balanced', 'unbalanced', 'complex']
default_rearrangement_type_colors = seaborn.color_palette('Dark2', len(default_rearrangement_types))


def chromosome_type_plot(ax, breakends, bin_size=20000000, rearrangement_types=None, full_genome=True):
    if rearrangement_types is None:
        rearrangement_types = default_rearrangement_types
        rearrangement_type_colors = default_rearrangement_type_colors
    else:
        rearrangement_type_colors = seaborn.color_palette('Dark2', len(rearrangement_types))

    breakends = breakends.loc[(breakends['chromosome'].isin(refgenome.info.chromosomes))].copy()

    breakends.set_index('chromosome', inplace=True)
    breakends['chromosome_start'] = refgenome.info.chromosome_start
    breakends.reset_index(inplace=True)

    breakends['plot_position'] = breakends['position'] + breakends['chromosome_start']
    breakends['plot_bin'] = breakends['plot_position'] / bin_size
    breakends['plot_bin'] = breakends['plot_bin'].astype(int)

    if full_genome:
        plot_bin_starts = np.arange(0, refgenome.info.chromosome_end.max(), bin_size)
    else:
        min_x, max_x = breakends['plot_position'].min(), breakends['plot_position'].max()
        plot_bin_starts = np.arange(min_x, max_x, bin_size)
    plot_bins = (plot_bin_starts / bin_size).astype(int)

    breakends = breakends[['prediction_id', 'plot_bin', 'rearrangement_type']].drop_duplicates()

    count_table = breakends.groupby(['plot_bin', 'rearrangement_type']).size().unstack()
    count_table = count_table.reindex(plot_bins).fillna(0)

    accum_counts = None
    for idx, rearrangement_type in enumerate(rearrangement_types):
        if rearrangement_type not in count_table.columns:
            continue
        ax.bar(plot_bin_starts, count_table[rearrangement_type].values,
               bottom=accum_counts, width=bin_size,
               facecolor=rearrangement_type_colors[idx],
               edgecolor=rearrangement_type_colors[idx])
        if accum_counts is None:
            accum_counts = count_table[rearrangement_type].values
        else:
            accum_counts += count_table[rearrangement_type].values

    if full_genome:
        ax.set_xlim((-0.5, refgenome.info.chromosome_end.max()))
    else:
        ax.set_xlim((min_x, max_x))
    ax.set_xlabel('chromosome')
    ax.set_xticks([0] + list(refgenome.info.chromosome_end.values))
    ax.set_xticklabels([])
    ax.set_ylabel('count')
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(refgenome.info.chromosome_mid))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(refgenome.info.chromosomes))
    ax.xaxis.grid(False, which="minor")


def chromosome_type_plot_legend(ax, rearrangement_types=None):
    if rearrangement_types is None:
        rearrangement_types = default_rearrangement_types
        rearrangement_type_colors = default_rearrangement_type_colors
    else:
        rearrangement_type_colors = seaborn.color_palette('Dark2', len(rearrangement_types))

    ax.legend([plt.Circle((0, 0), color=c) for c in rearrangement_type_colors],
              list(rearrangement_types), loc="upper right")


def type_size_plot(data, type_col='type'):
    data = data.copy()
    data['type'] = data['type'].astype('category')
    data['rearrangement_type'] = pd.Categorical(
        data['rearrangement_type'],
        categories=[
            'duplication',
            'inversion',
            'balanced',
            'foldback',
            'deletion',
            'unbalanced',
        ])
    data['type'] = pd.Categorical(
        data['type'],
        categories=[
            'duplication',
            'inversion',
            'deletion',
            'translocation',
        ])

    seaborn.factorplot(
        x=type_col, hue='size_class', kind='count',
        data=data, sharex=False,
        aspect=3, height=2.5)
    seaborn.despine(trim=True)


def breakpoint_adjacent_density_plot(ax, breakends):
    breakends = breakends.drop_duplicates(['prediction_id', 'prediction_side'])

    breakends = breakends.loc[(breakends['chromosome'].isin(refgenome.info.chromosomes))]

    breakends.set_index('chromosome', inplace=True)
    breakends['chromosome_start'] = refgenome.info.chromosome_start
    breakends['chromosome_color'] = pd.Series(plots.colors.create_chromosome_color_map())
    breakends.reset_index(inplace=True)

    breakends['plot_position'] = breakends['position'] + breakends['chromosome_start']

    assert not breakends['chromosome_color'].isnull().any()

    ax.scatter(
        breakends['plot_position'], breakends['adjacent_density'],
        facecolors=list(breakends['chromosome_color']),
        edgecolors=list(breakends['chromosome_color']),
        alpha=1.0, s=10, lw=0
    )

    ax.set_xlim((-0.5, refgenome.info.chromosome_end.max()))
    ax.set_xlabel('chromosome')
    ax.set_xticks([0] + list(refgenome.info.chromosome_end.values))
    ax.set_xticklabels([])
    ax.set_ylabel('count')
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(refgenome.info.chromosome_mid))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(refgenome.info.chromosomes))
    ax.xaxis.grid(False, which="minor")

    ax.set_ylim(-.05 * breakends['adjacent_density'].max(), 1.1 * breakends['adjacent_density'].max())
    ax.set_ylabel('breakpoint density')


def plot_polar_arc(ax, u1, u2, r, rt, color='k', lw=0.5):
    '''
    Plot an arc in polar coordinates
    '''

    if u1 == u2:
        u2 += np.finfo(float).eps

    theta_1 = u1 * 2. * np.pi
    theta_2 = u2 * 2. * np.pi

    x1 = r * np.cos(theta_1)
    y1 = r * np.sin(theta_1)

    x2 = r * np.cos(theta_2)
    y2 = r * np.sin(theta_2)

    xt = rt * np.cos(0.5 * (theta_1 + theta_2))
    yt = rt * np.sin(0.5 * (theta_1 + theta_2))

    m1 = (y1 - yt) / (x1 - xt)
    m2 = (y2 - yt) / (x2 - xt)

    a = np.array([
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1],
                  [1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1],
                  [0, m1, 0, 0, -1, 0],
                  [2*m2, m2, 0, -2, -1, 0]
                 ])

    b = np.array([x1, y1, x2, y2, 0, 0])

    try:
        c = np.linalg.solve(a, b)
    except:
        print(u1, u2, r, rt)
        raise

    xs = list()
    ys = list()

    for t in np.linspace(0, 1, 101):
        basis = np.array([t**2, t, 1])
        x = np.dot(c[:3], basis)
        y = np.dot(c[3:], basis)
        xs.append(x)
        ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)
    
    r = np.sqrt(xs**2 + ys**2)
    theta = np.arctan(ys / xs)
    theta[xs < 0] = theta[xs < 0] + np.pi

    ax.plot(theta, r, color=color, lw=lw)


def breakpoint_circos_plot(ax, breakpoints):
    genome_length = refgenome.info.chromosome_lengths.sum()

    for breakend in ('1', '2'):
        breakpoints.set_index('chromosome_' + breakend, inplace=True)
        breakpoints['chromosome_' + breakend + '_start'] = refgenome.info.chromosome_start
        breakpoints.reset_index(inplace=True)

        breakpoints['plot_position_' + breakend] = breakpoints['position_' + breakend] + breakpoints['chromosome_' + breakend + '_start']
        breakpoints['plot_fraction_' + breakend] = breakpoints['plot_position_' + breakend] / genome_length

    for idx in breakpoints.index:
        plot_polar_arc(ax, breakpoints.loc[idx, 'plot_fraction_1'], breakpoints.loc[idx, 'plot_fraction_2'], 1., 0.2)

    chrom_colors = plots.colors.create_chromosome_color_map()
    for chrom, start, end in zip(refgenome.info.chromosomes, refgenome.info.chromosome_start, refgenome.info.chromosome_end):
        start = 2. * np.pi * start / genome_length
        end = 2. * np.pi * end / genome_length
        ax.plot(np.linspace(start, end, 1001), [1.] * 1001, lw=20, color=chrom_colors[chrom], solid_capstyle='butt')

    xticks = 2. * np.pi * np.array([0] + list(refgenome.info.chromosome_end.values)) / genome_length
    tickmids = 2. * np.pi * refgenome.info.chromosome_mid / genome_length

    ax.set_xlabel('chromosome')
    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(tickmids))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(refgenome.info.chromosomes))
    ax.xaxis.grid(False)

    ax.set_ylim(0., 1.)
    ax.set_ylabel('')


def plot_breakends(ax, breakends, chromosome=None, lw=0.5):
    """ Plot breakpoint flags and arcs on a genome axis

    Args:
        ax: matplotlib axis
        breakends: breakend table
    
    Kwargs:
        lw: line width for arcs

    The breakends table should have the following columns:
        - prediction_id
        - chromosome
        - position
        - strand

    """
    chromosome_info = refgenome.info.chromosome_info[['chr', 'chromosome_start']].copy()
    chromosome_info['chromosome'] = chromosome_info['chr']

    if chromosome is None:
        plot_data = breakends.merge(chromosome_info)
        plot_data = plot_data[plot_data['chr'].isin(refgenome.info.chromosomes)]
        plot_data['plot_position'] = plot_data['position'] + plot_data['chromosome_start']

    else:
        plot_data = breakends.copy()
        plot_data['plot_position'] = plot_data['position']

    xlim = ax.get_xlim()
    xrng = xlim[1] - xlim[0]

    ylim = ax.get_ylim()
    yrng = ylim[1] - ylim[0]

    arrow_length = 0.01 * xrng

    head_width = yrng / 30.
    head_length = xrng / 100.

    prediction_ids = list(plot_data['prediction_id'].unique())
    random.shuffle(prediction_ids)
    color_palette = seaborn.color_palette('hls', len(prediction_ids))
    prediction_colors = dict(zip(prediction_ids, color_palette))

    for idx in plot_data.index:
        if chromosome is not None and plot_data.loc[idx, 'chromosome'] != chromosome:
            continue
        prediction_id = plot_data.loc[idx, 'prediction_id']
        x = plot_data.loc[idx, 'plot_position']
        strand = plot_data.loc[idx, 'strand']
        y = np.random.uniform(ylim[0] + 0.75*yrng, ylim[0] + 0.95*yrng)
        offset = (arrow_length, -arrow_length)[strand == '+']
        color = prediction_colors[prediction_id]
        ax.arrow(x, y, offset, 0, color=color, lw=lw, alpha=1.0, head_width=head_width, head_length=head_length)
        ax.plot([x, x], [ylim[0], ylim[1]], color=color, lw=lw, ls='-', alpha=1.0)

    for prediction_id, pred_breakends in plot_data.groupby('prediction_id'):
        if len(pred_breakends.index) != 2:
            continue
        if chromosome is not None and set(pred_breakends['chromosome'].values) != set([chromosome]):
            continue
        pos1 = pred_breakends['plot_position'].min()
        pos2 = pred_breakends['plot_position'].max()
        posmid = (pos1 + pos2) / 2.
        height = 0.5 * yrng * (pos2 - pos1) / xrng
        ypos = ylim[1]
        yposmid = ypos + height
        spl = scipy.interpolate.make_interp_spline([pos1, posmid, pos2], [ypos, yposmid, ypos], k=2)
        pos = np.linspace(pos1, pos2, 100)
        color = prediction_colors[prediction_id]
        ax.plot(pos, spl(pos), ls='-', lw=lw, color=color)

    ax.set_ylim((ylim[0], ylim[1] + 0.5 * (ylim[1] - ylim[0])))


