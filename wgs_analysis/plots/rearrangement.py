'''
# Rearrangement plotting

'''

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn

import ith_project.analysis.plots as plots
import ith_project.analysis.refgenome as refgenome
import ith_project.analysis.plots.colors


rearrangement_types = ['deletion', 'duplication', 'inversion', 'balanced', 'unbalanced', 'complex']
rearrangement_type_colors = seaborn.color_palette("Set1", len(rearrangement_types))


def create_breakends(breakpoints, data_cols=[]):
    breakends = breakpoints[['prediction_id', 'chromosome_1', 'strand_1', 'position_1', 'chromosome_2', 'strand_2', 'position_2']].copy()
    breakends.set_index('prediction_id', inplace=True)
    breakends.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in breakends.columns])
    breakends = breakends.stack()
    breakends.index.names = ('prediction_id', 'prediction_side')
    breakends = breakends.reset_index()
    breakends['prediction_side'] = np.where(breakends['prediction_side'] == '1', 0, 1)
    breakends = breakends.merge(breakpoints[['prediction_id'] + data_cols], on='prediction_id')
    return breakends


def chromosome_type_plot(ax, breakends, bin_size=20000000):
    breakends = breakends.loc[(breakends['chromosome'].isin(refgenome.info.chromosomes))].copy()

    breakends.set_index('chromosome', inplace=True)
    breakends['chromosome_start'] = refgenome.info.chromosome_start
    breakends.reset_index(inplace=True)

    breakends['plot_position'] = breakends['position'] + breakends['chromosome_start']
    breakends['plot_bin'] = breakends['plot_position'] / bin_size
    breakends['plot_bin'] = breakends['plot_bin'].astype(int)

    plot_bin_starts = np.arange(0, refgenome.info.chromosome_end.max(), bin_size)
    plot_bins = (plot_bin_starts / bin_size).astype(int)

    breakends = breakends[['prediction_id', 'plot_bin', 'rearrangement_type']].drop_duplicates()

    count_table = breakends.groupby(['plot_bin', 'rearrangement_type']).size().unstack()
    count_table = count_table.reindex(plot_bins).fillna(0)

    bar_colors = seaborn.color_palette("Set1", len(rearrangement_types))

    accum_counts = None
    for idx, rearrangement_type in enumerate(rearrangement_types):
        if rearrangement_type not in count_table.columns:
            continue
        ax.bar(plot_bin_starts, count_table[rearrangement_type].values,
               bottom=accum_counts, width=bin_size,
               facecolor=bar_colors[idx], edgecolor=bar_colors[idx])
        if accum_counts is None:
            accum_counts = count_table[rearrangement_type].values
        else:
            accum_counts += count_table[rearrangement_type].values

    ax.set_xlim((-0.5, refgenome.info.chromosome_end.max()))
    ax.set_xlabel('chromosome')
    ax.set_xticks([0] + list(refgenome.info.chromosome_end.values))
    ax.set_xticklabels([])
    ax.set_ylabel('count')
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(refgenome.info.chromosome_mid))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(refgenome.info.chromosomes))
    ax.xaxis.grid(False, which="minor")


def chromosome_type_plot_legend(ax):

    ax.legend([plt.Circle((0, 0), color=c) for c in rearrangement_type_colors],
              list(rearrangement_types), loc="upper right")


def breakpoint_adjacent_density_plot(ax, breakends):
    breakends = breakends.drop_duplicates(['prediction_id', 'prediction_side'])

    breakends = breakends.loc[(breakends['chrom'].isin(refgenome.info.chromosomes))]

    breakends.set_index('chrom', inplace=True)
    breakends['chromosome_start'] = refgenome.info.chromosome_start
    breakends['chromosome_color'] = pd.Series(plots.colors.create_chromosome_color_map())
    breakends.reset_index(inplace=True)

    breakends['plot_coord'] = breakends['coord'] + breakends['chromosome_start']

    assert not breakends['chromosome_color'].isnull().any()

    ax.scatter(
        breakends['plot_coord'], breakends['adjacent_density'],
        facecolors=list(breakends['chromosome_color']),
        edgecolors=list(breakends['chromosome_color']),
        alpha=0.5, s=5, lw=0
    )

    ax.set_xlim(min(breakends['plot_coord']), max(breakends['plot_coord']))
    ax.set_xticks(refgenome.info.chromosome_mid, minor=False)
    ax.set_xticklabels(refgenome.info.chromosomes, minor=False)
    ax.set_xticks(refgenome.info.chromosome_end, minor=True)
    ax.set_xticklabels([], minor=True)
    ax.grid(False, which='major')

    ax.set_ylim(-0.0001, 1.1 * breakends['adjacent_density'].max())
    ax.set_ylabel('breakpoint density')


def plot_polar_arc(ax, u1, u2, r, rt):
    '''
    Plot an arc in polar coordinates
    '''

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

    c = np.linalg.solve(a, b)

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

    ax.plot(theta, r)

