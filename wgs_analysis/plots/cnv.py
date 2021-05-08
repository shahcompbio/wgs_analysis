'''
# Copy number plotting

Plotting functions for displaying copy number taking tables of copy number predictions as input.

'''

import math
import itertools
import collections
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import seaborn

import wgs_analysis.plots.colors
import wgs_analysis.plots.positions
import wgs_analysis.plots.utils
import wgs_analysis.algorithms as algorithms
import wgs_analysis.algorithms.cnv
import wgs_analysis.refgenome
import wgs_analysis.algorithms.merge


def create_segments(df, field):
    segments = np.array([[df['start'].values, df[field].values], [df['end'].values, df[field].values]])
    segments = np.transpose(segments, (2, 0, 1))
    return segments


def create_connectors(df, field):
    prev = df.iloc[:-1].reset_index()
    next = df.iloc[1:].reset_index()
    mids = ((prev[field] + next[field]) / 2.0).values
    prev_cnct = np.array([[prev['end'].values, prev[field].values], [prev['end'].values, mids]])
    prev_cnct = np.transpose(prev_cnct, (2, 0, 1))
    next_cnct = np.array([[next['start'].values, mids], [next['start'].values, next[field].values]])
    next_cnct = np.transpose(next_cnct, (2, 0, 1))
    return np.concatenate([prev_cnct, next_cnct])


def create_quads(df, field):
    quads = np.array([
        [df['start'].values, np.zeros(len(df.index))],
        [df['start'].values, df[field].values],
        [df['end'].values, df[field].values],
        [df['end'].values, np.zeros(len(df.index))],
    ])
    quads = np.transpose(quads, (2, 0, 1))
    return quads


def plot_segments(ax, cnv, value_col, color, fill=False, fill_alpha=0.5, lw=1):
    """
    Plot segment copy number as line plots

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): cnv table
        value_col (str): column name to plot
        color (str or tuple): color of lines

    Plot segment copy number as line plots.  The columns 'start' and 'end'
    are expected and should be adjusted for full genome plots.  Values from
    the column given by 'value_col' are plotted with color given by 'color'.

    """ 

    cnv = cnv.sort_values('start')

    segments = create_segments(cnv, value_col)
    ax.add_collection(matplotlib.collections.LineCollection(segments, colors=color, lw=lw))

    connectors = create_connectors(cnv, value_col)
    ax.add_collection(matplotlib.collections.LineCollection(connectors, colors=color, lw=lw))

    if fill:
        quad_color = colorConverter.to_rgba(color, alpha=fill_alpha)
        quads = create_quads(cnv, value_col)
        ax.add_collection(matplotlib.collections.PolyCollection(quads, facecolors=quad_color, edgecolors=quad_color, lw=0))


def plot_cnv_segments(ax, cnv, column, segment_color, fill=False):
    """ Plot raw copy number as line plots

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): cnv table
        column (str): name of copies column
        segment_color: color of the segments

    Plot copy number as line plots.  The columns 'start' and 'end'
    are expected and should be adjusted for full genome plots.  Values from the
    'column' columns are plotted.

    """ 

    quad_color = colorConverter.to_rgba(segment_color, alpha=0.5)

    cnv = cnv.sort_values('start')

    segments = create_segments(cnv, column)
    ax.add_collection(matplotlib.collections.LineCollection(segments, colors=segment_color, lw=1))

    connectors = create_connectors(cnv, column)
    ax.add_collection(matplotlib.collections.LineCollection(connectors, colors=segment_color, lw=1))

    if fill:
        quads = create_quads(cnv, column)
        ax.add_collection(matplotlib.collections.PolyCollection(quads, facecolors=quad_color, edgecolors=quad_color, lw=0))


def plot_cnv_genome(ax, cnv, maxcopies=4, minlength=1000, major_col='major_raw', minor_col='minor_raw', scatter=False, squashy=False):
    """
    Plot major/minor copy number across the genome

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): `cnv_site` table
        maxcopies (int): maximum number of copies for setting y limits
        minlength (int): minimum length of segments to be drawn
        major_col (str): name of column to use as major copy number
        minor_col (str): name of column to use as minor copy number
        scatter (boolean): display segments as scatter points not segments
        squashy (boolean): squash the y axis to display all copy numbers

    """

    segment_color_major = plt.get_cmap('RdBu')(0.1)
    segment_color_minor = plt.get_cmap('RdBu')(0.9)

    cnv = cnv.copy()

    squash_coeff = 0.15
    squash_f = lambda a: np.tanh(squash_coeff * a)
    if squashy:
        cnv[major_col] = squash_f(cnv[major_col])
        cnv[minor_col] = squash_f(cnv[minor_col])

    if 'length' not in cnv:
        cnv['length'] = cnv['end'] - cnv['start']

    cnv = cnv[['chromosome', 'start', 'end', 'length', major_col, minor_col]]

    cnv = cnv[cnv['length'] >= minlength]
    
    cnv = cnv[cnv['chromosome'].isin(wgs_analysis.refgenome.info.chromosomes)]

    cnv.set_index('chromosome', inplace=True)
    cnv['chromosome_start'] = wgs_analysis.refgenome.info.chromosome_start
    cnv.reset_index(inplace=True)

    cnv['start'] = cnv['start'] + cnv['chromosome_start']
    cnv['end'] = cnv['end'] + cnv['chromosome_start']

    if scatter:
        cnv['mid'] = 0.5 * (cnv['start'] + cnv['end'])
        for column, color in ((minor_col, segment_color_minor), (major_col, segment_color_major)):
            clipped_cnv = cnv[cnv[column] < maxcopies]
            amp_cnv = cnv[cnv[column] >= maxcopies]
            ax.scatter(clipped_cnv['mid'], clipped_cnv[column], color=color, s=2, alpha=1)
            ax.scatter(clipped_cnv['mid'], clipped_cnv[column], color=color, s=10, alpha=0.1)
            ax.scatter(amp_cnv['mid'], np.ones(amp_cnv.shape[0]) * maxcopies, color=color, s=30)

    else:
        plot_cnv_segments(ax, cnv, column=major_col, segment_color=segment_color_major)
        plot_cnv_segments(ax, cnv, column=minor_col, segment_color=segment_color_minor)

    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim((-0.5, wgs_analysis.refgenome.info.chromosome_end.max()))
    ax.set_xlabel('chromosome')
    ax.set_xticks([0] + list(wgs_analysis.refgenome.info.chromosome_end.values))
    ax.set_xticklabels([])

    if squashy:
        yticks = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 20])
        yticks_squashed = squash_f(yticks)
        ytick_labels = [str(a) for a in yticks]
        ax.set_yticks(yticks_squashed)
        ax.set_yticklabels(ytick_labels)
        ax.set_ylim((-0.01, 1.01))
        ax.spines['left'].set_bounds(0, 1)
    else:
        ax.set_ylim((-0.05*maxcopies, maxcopies))
        ax.set_yticks(range(0, int(maxcopies) + 1))
        ax.spines['left'].set_bounds(0, maxcopies)

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(wgs_analysis.refgenome.info.chromosome_mid))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(wgs_analysis.refgenome.info.chromosomes))

    ax.xaxis.grid(True, which='major', linestyle=':')
    ax.yaxis.grid(True, which='major', linestyle=':')


def plot_cnv_chromosome(ax, cnv, chromosome, start=None, end=None, maxcopies=4, minlength=1000, major_col='major_raw', minor_col='minor_raw', fontsize=None, scatter=False, squashy=False, squash_coeff=None, grid=True, yticks=None):
    """
    Plot major/minor copy number across a chromosome

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): `cnv_site` table
        chromosome (str): chromosome to plot
        start (int): start of range in chromosome to plot
        end (int): start of range in chromosome to plot
        maxcopies(float): maximum number of copies for setting y limits
        minlength(int): minimum length of segments to be drawn
        major_col (str): name of column to use as major copy number
        minor_col (str): name of column to use as minor copy number
        fontsize (int): size of x tick labels
        scatter (boolean): display segments as scatter points not segments
        squashy (boolean): squash the y axis to display all copy numbers
        squash_coeff (float): coefficient for squashing y axis
        grid (boolean): whether to display grid on plot
        yticks (array): override y ticks

    """

    chromosome_length = wgs_analysis.refgenome.info.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_length']

    segment_color_major = plt.get_cmap('RdBu')(0.1)
    segment_color_minor = plt.get_cmap('RdBu')(0.9)

    cnv = cnv.copy()

    if squash_coeff is None:
        squash_coeff = 0.15
    squash_f = lambda a: np.tanh(squash_coeff * a)
    if squashy:
        cnv[major_col] = squash_f(cnv[major_col])
        cnv[minor_col] = squash_f(cnv[minor_col])

    cnv = cnv.loc[(cnv['chromosome'] == chromosome)]
    cnv = cnv.loc[(cnv['length'] >= minlength)]

    if start is None:
        start = 0
    if end is None:
        end = chromosome_length

    cnv['genomic_length'] = cnv['end'] - cnv['start']
    cnv['length_fraction'] = cnv['length'].astype(float) / cnv['genomic_length'].astype(float)
    cnv = cnv.loc[(cnv['length_fraction'] >= 0.5)]

    cnv = cnv.loc[(cnv['end'] > start)]
    cnv = cnv.loc[(cnv['start'] < end)]
    
    if scatter:
        cnv['mid'] = 0.5 * (cnv['start'] + cnv['end'])
        for column, color in ((minor_col, segment_color_minor), (major_col, segment_color_major)):
            clipped_cnv = cnv[cnv[column] < maxcopies]
            amp_cnv = cnv[cnv[column] >= maxcopies]
            ax.scatter(clipped_cnv['mid'], clipped_cnv[column], color=color, s=2, alpha=1)
            ax.scatter(clipped_cnv['mid'], clipped_cnv[column], color=color, s=10, alpha=0.1)
            ax.scatter(amp_cnv['mid'], np.ones(amp_cnv.shape[0]) * maxcopies, color=color, s=30)

    else:
        plot_cnv_segments(ax, cnv, column=major_col, segment_color=segment_color_major)
        plot_cnv_segments(ax, cnv, column=minor_col, segment_color=segment_color_minor)

    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    xticks = np.arange(0, chromosome_length, 2e7)
    xticklabels = ['{0:d}M'.format(int(x / 1e6)) for x in xticks]
    xminorticks = np.arange(0, chromosome_length, 1e6)
    ax.set_xlabel(f'chromosome {chromosome}')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(xminorticks))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim((start, end))

    if squashy:
        if yticks is None:
            yticks = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 20])
        yticks_squashed = squash_f(np.array(yticks))
        ytick_labels = [str(a) for a in yticks]
        ax.set_yticks(yticks_squashed)
        ax.set_yticklabels(ytick_labels)
        ax.set_ylim((-0.01, 1.01))
        ax.spines['left'].set_bounds(0, 1)
    else:
        if yticks is None:
            yticks = range(0, int(maxcopies) + 1, 2)
        ax.set_ylim((-0.05*maxcopies, maxcopies))
        ax.set_yticks(yticks)
        ax.spines['left'].set_bounds(0, maxcopies)
  
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    wgs_analysis.plots.utils.trim_spines_to_ticks(ax)

    if grid:
        ax.xaxis.grid(True, which='major', linestyle=':')
        ax.yaxis.grid(True, which='major', linestyle=':')


def plot_cnv(cnv, site_ids, chromosome, start=None, end=None, maxcopies=2):
    """
    Plot major/minor copy number across a chromosome and breakends as flags

    Args:
        cnv (pandas.DataFrame): `cnv_site` table
        site_ids (str): sites to plot
        chromosome (str): chromosome to plot
        start (int): start of range in chromosome to plot
        end (int): start of range in chromosome to plot
        maxcopies(int): maximum number of copies for setting y limits

    """
    
    site_axes = collections.OrderedDict()
    
    num_sites = len(site_ids)

    for fig_idx, sample_id in enumerate(site_ids):
        
        ax = plots.utils.setup_axes(plt.subplot(num_sites, 1, fig_idx+1))
        
        plots.cnv.plot_cnv_chromosome(ax, cnv, sample_id, chromosome, start=start, end=end, maxcopies=maxcopies)

        if fig_idx != num_sites - 1:
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_ticks_position('none')

        site_axes[sample_id] = ax
        
    plt.tight_layout()
    
    return site_axes


def plot_cnv_brks(cnv, brks, site_ids, chromosome, start=None, end=None, maxcopies=2):
    """
    Plot major/minor copy number across a chromosome and breakends as flags

    Args:
        cnv (pandas.DataFrame): `cnv_site` table
        brks (pandas.DataFrame): `breakend_site` table
        site_ids (str): sites to plot
        chromosome (str): chromosome to plot
        start (int): start of range in chromosome to plot
        end (int): start of range in chromosome to plot
        maxcopies(int): maximum number of copies for setting y limits

    """
    
    site_axes = collections.OrderedDict()
    
    num_sites = len(site_ids)

    for fig_idx, sample_id in enumerate(site_ids):
        
        site_brks = brks[brks['sample_id'] == sample_id]
        
        ax = plots.utils.setup_axes(plt.subplot(num_sites, 1, fig_idx+1))
        
        plots.cnv.plot_cnv_chromosome(ax, cnv, sample_id, chromosome, start=start, end=end, maxcopies=maxcopies)
        plots.positions.plot_breakends(ax, brks, sample_id, chromosome, start=start, end=end)

        if fig_idx != num_sites - 1:
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_ticks_position('none')

        site_axes[sample_id] = ax
        
    plt.tight_layout()
    
    return site_axes


def plot_cnv_scatter(ax, cnv, site, chromosome_color, point_alpha=1.0, line_alpha=1.0):

    cnv = cnv[cnv['sample_id'] == site]

    cnv = cnv[cnv['length'] >= 10000]
    
    cnv = cnv.set_index(['chromosome', 'start', 'end'])

    cs = [chromosome_color[c[0]] for c in cnv.index.values]
    sz = cnv['length'] / 100000.

    points = ax.scatter(cnv['major_raw'], cnv['minor_raw'], s=sz, facecolor=cs, edgecolor=cs, linewidth=0.0, zorder=2, alpha=point_alpha)

    ax.set_xlim((-0.5, cnv['major_raw'].quantile(0.95) + 0.5))
    ax.set_ylim((-0.5, cnv['minor_raw'].quantile(0.95) + 0.5))


def plot_cnv_scatter_pairwise(ax, cnv, site_a, site_b, chromosome_color, point_alpha=1.0, line_alpha=1.0):

    cnv = cnv.set_index(['chromosome', 'start', 'end', 'sample_id']).unstack()
    cnv = cnv.fillna(0.0)

    cs = [chromosome_color[c[0]] for c in cnv.index.values]
    sz = (cnv['length'][site_a] + cnv['length'][site_b])/500000.

    for x1, y1, x2, y2, c, s in zip(cnv['major_raw'][site_a], cnv['minor_raw'][site_a], cnv['major_raw'][site_b], cnv['minor_raw'][site_b], cs, sz):
        lines = plt.plot([x1, x2], [y1, y2], alpha=line_alpha, zorder=1, color=c, lw=1.0)

    points1 = ax.scatter(cnv['major_raw'][site_a], cnv['minor_raw'][site_a], s=sz, facecolor=cs, edgecolor=cs, linewidth=0.0, zorder=2, alpha=point_alpha)
    points2 = ax.scatter(cnv['major_raw'][site_b], cnv['minor_raw'][site_b], s=sz, facecolor=cs, edgecolor=cs, linewidth=0.0, zorder=2, alpha=point_alpha)


def plot_patient_cnv(cnv):

    sites = cnv['sample_id'].unique()

    plots.utils.setup_plot()
    fig = plt.figure(figsize=(12,5*len(sites)))

    chromosome_color = plots.colors.create_chromosome_color_map()

    for idx, sample_id in enumerate(sites):
        ax = plots.utils.setup_axes(plt.subplot(len(sites), 1, idx+1))
        plot_cnv_scatter(ax, cnv, sample_id, chromosome_color)
        if idx == len(sites) - 1:
            ax.set_xlabel('major copy number')
            ax.set_ylabel('minor copy number')
        ax.set_title(helpers.plot_ids[sample_id] + ' copies')

    fig.legend([plt.Circle((0, 0), radius=1, color=color) for chrom, color in chromosome_color.items()], chromosome_color.keys(), loc="lower right")

    plt.tight_layout()

    return fig


def plot_cnv_pairwise(ax, cnv, site_a, site_b, chromosome_color, highlighted_chromosomes=None):

    cnv = cnv[(cnv['sample_id'] == site_a) | (cnv['sample_id'] == site_b)]
    
    cnv = algorithms.cnv.cluster_copies(cnv)
    
    cnv = cnv[cnv['length'] >= 5000000]
    
    if highlighted_chromosomes is None:
        plot_cnv_scatter_pairwise(ax, cnv, site_a, site_b, chromosome_color, 1.0, 0.5)
    else:
        plot_cnv_scatter_pairwise(ax, cnv[~(cnv['chromosome'].isin(highlighted_chromosomes))], site_a, site_b, chromosome_color, 0.1, 0.1)
        plot_cnv_scatter_pairwise(ax, cnv[cnv['chromosome'].isin(highlighted_chromosomes)], site_a, site_b, chromosome_color, 1.0, 1.0)

    ax.grid(True)


def plot_patient_cnv_pairwise(fig, cnv, highlighted_chromosomes=None):

    pairs = list(itertools.combinations(cnv['sample_id'].unique(), 2))

    chromosome_color = plots.colors.create_chromosome_color_map()

    subplot_width = np.ceil(np.sqrt(len(pairs)))
    subplot_height = np.ceil(len(pairs) / subplot_width)

    for idx, (site_a, site_b) in enumerate(pairs):
        ax = plots.utils.setup_axes(plt.subplot(subplot_height, subplot_width, idx+1))
        plot_cnv_pairwise(ax, cnv, site_a, site_b, chromosome_color, highlighted_chromosomes)
        if idx == subplot_height * (subplot_width - 1):
            ax.set_xlabel('major copy number')
            ax.set_ylabel('minor copy number')
        ax.set_title(helpers.plot_ids[site_a] + ' vs ' + helpers.plot_ids[site_b])

    fig.legend([plt.Circle((0, 0), radius=1, color=color) for chrom, color in chromosome_color.items()], chromosome_color.keys(), loc="lower right")

    plt.tight_layout()


def create_uniform_segments(segment_length, max_end):
    """
    Create a table of uniform segments with a given segment length

    Args:
        segment_length (int): uniform segment length
        max_end (int): maximum end position of segments

    Returns:
        pandas.DataFrame: table of uniform segments

    The returned table will have columns start, end.  Segments are created according
    to given max end.
    """

    num_segments = np.ceil(float(max_end) / float(segment_length))

    starts = np.arange(0, num_segments*segment_length, segment_length)
    starts = starts.astype(int) + 1

    segments = pd.DataFrame({'start':starts})
    segments['end'] = segments['start'] + segment_length - 1

    return segments


def create_uniform_segments_genome(segment_length):
    """
    Create a table of uniform segments with a given segment length

    Args:
        segment_length (int): uniform segment length

    Returns:
        pandas.DataFrame: table of uniform segments

    The returned table will have columns chromosome, start, end.  Segments are created according
    to chromosome lengths.
    """

    num_segments = (wgs_analysis.refgenome.info.chromosome_lengths.astype(float) / segment_length).apply(np.ceil)

    chroms = np.concatenate([np.repeat(c, n) for c, n in num_segments.iteritems()])
    starts = np.concatenate([np.arange(0, m*segment_length, segment_length) for c, m in num_segments.iteritems()])

    chroms = chroms.astype(str)
    starts = starts.astype(int) + 1

    segments = pd.DataFrame({'chromosome':chroms, 'start':starts})
    segments['end'] = segments['start'] + segment_length - 1

    return segments


def uniform_resegment(cnv, segment_length=100000):
    """
    Create a table of uniform segments data from arbitrary segments segment data

    Args:
        cnv (pandas.DataFrame): segment data
        segment_length (int): uniform segment length

    Returns:
        pandas.DataFrame: resegmented table

    The cnv table should have columns start, end.  Returns a resegmented
    table with columns start, end, start_reseg, end_reseg.  Original 
    rows will be represented multiple times if they are split by uniform segments,
    and can be identified by having the same start, end.  The start_reseg
    and end_reseg columns represent the subsegment resulting from the split.
    """

    # First segment cannot start at coordinate less than 1
    assert cnv['start'].min() >= 1

    cnv['seg_idx'] = xrange(len(cnv.index))

    uniform_segments = create_uniform_segments(segment_length, cnv['end'].max())
    uniform_segments['reseg_idx'] = xrange(len(uniform_segments.index))

    # Find starts of the resegmentation that fall within cnv segments
    seg_idx_1, reseg_idx_1 = wgs_analysis.algorithms.merge.interval_position_overlap_unsorted(
        cnv[['start', 'end']].values,
        uniform_segments['start'].values)
    reseg_1 = pd.DataFrame({'seg_idx': seg_idx_1, 'reseg_idx': reseg_idx_1})

    # Find ends of the resegmentation that fall within cnv segments
    seg_idx_2, reseg_idx_2 = wgs_analysis.algorithms.merge.interval_position_overlap_unsorted(
        cnv[['start', 'end']].values,
        uniform_segments['end'].values)
    reseg_2 = pd.DataFrame({'seg_idx': seg_idx_2, 'reseg_idx': reseg_idx_2})

    # Find starts of the cnv segments that fall within the resegmentation
    reseg_idx_3, seg_idx_3 = wgs_analysis.algorithms.merge.interval_position_overlap_unsorted(
        uniform_segments[['start', 'end']].values,
        cnv['start'].values)
    reseg_3 = pd.DataFrame({'seg_idx': seg_idx_3, 'reseg_idx': reseg_idx_3})

    reseg = pd.concat([reseg_1, reseg_2, reseg_3], ignore_index=True).drop_duplicates()

    cnv = cnv.merge(reseg, on='seg_idx')
    cnv = cnv.merge(uniform_segments, on='reseg_idx', suffixes=('', '_reseg'))

    assert np.all(
        ((cnv['start'] >= cnv['start_reseg']) & (cnv['start'] <= cnv['end_reseg'])) |
        ((cnv['end'] >= cnv['start_reseg']) & (cnv['end'] <= cnv['end_reseg'])) |
        ((cnv['start'] <= cnv['start_reseg']) & (cnv['end'] >= cnv['end_reseg'])))

    return cnv


def uniform_resegment_genome(cnv, segment_length=100000):
    """
    Create a table of uniform segments data from arbitrary segments segment data

    Args:
        cnv (pandas.DataFrame): segment data
        segment_length (int): uniform segment length

    Returns:
        pandas.DataFrame: resegmented table

    The cnv table should have columns chromosome, start, end.  Returns a resegmented
    table with columns chromosome, start, end, start_reseg, end_reseg.  Original 
    rows will be represented multiple times if they are split by uniform segments,
    and can be identified by having the same chromosome, start, end.  The start_reseg
    and end_reseg columns represent the subsegment resulting from the split.
    """

    return cnv.groupby('chromosome').apply(lambda cnv: uniform_resegment(cnv, segment_length=100000))


def uniform_segment_copies(cnv, sample_column, data_columns, segment_length=100000):
    """
    Create a table of uniformly segmented data from arbitrarily segment data

    Args:
        cnv (pandas.DataFrame): segment data
        sample_column (list): sample id column
        data_columns (list): data columns to resegment
        segment_length (int): uniform segment length

    Returns:
        pandas.DataFrame: resegmented table

    The cnv table should have sample_column, chrom, start and end columns in addition to
    the columns for which resegmentation is requested.  Returns a resegmented table
    with sample_column, chrom, segment_start, segment_end columns, in addition to the 
    requested columns calculated as length weighted averages of the original values.
    """

    cnv_reseg = uniform_resegment_genome(cnv[['chromosome', 'start', 'end']], segment_length=segment_length)

    cnv_reseg = cnv_reseg.merge(cnv, on=['chromosome', 'start', 'end'])

    cnv_reseg['segment_start'] = cnv_reseg['start_reseg'] / segment_length
    cnv_reseg['segment_start'] = cnv_reseg['segment_start'].astype(int) * segment_length + 1

    cnv_reseg.set_index([sample_column, 'chromosome', 'segment_start'], inplace=True)

    # Average requested columns weighted by length of segment
    # exclude null values in calculation
    for column in data_columns:

        # Length of resegmented segments
        cnv_reseg['length_reseg'] = cnv_reseg['end_reseg'] - cnv_reseg['start_reseg'] + 1

        # Mask segments with null values from normalization calculation
        cnv_reseg['length_reseg'] *= (cnv_reseg[column].notnull() * 1)

        # Normalize by total length of resegmented segments
        cnv_reseg['length_total_reseg'] = cnv_reseg.groupby(level=[0, 1, 2])['length_reseg'].sum()
        cnv_reseg['weight_reseg'] = (
            cnv_reseg['length_reseg'].astype(float) /
            cnv_reseg['length_total_reseg'].astype(float))
        cnv_reseg[column] *= cnv_reseg['weight_reseg']

        # Mask segments with null values from summation
        cnv_reseg[column] = cnv_reseg[column].fillna(0.0)

    cnv_reseg = cnv_reseg.groupby(level=[0, 1, 2])[data_columns].sum()

    cnv_reseg.reset_index(inplace=True)

    # Ensure the segments are consistent regardless of the cnv data
    seg_full = (
        create_uniform_segments_genome(segment_length)
        .rename(columns={'start':'segment_start'})
        .set_index(['chromosome', 'segment_start']))

    cnv_reseg = (
        cnv_reseg
        .set_index(['chromosome', 'segment_start', sample_column])
        .unstack()
        .reindex(seg_full.index)
        .stack(dropna=False)
        .reset_index())

    cnv_reseg['segment_end'] = cnv_reseg['segment_start'] + segment_length - 1

    return cnv_reseg


def plot_loh(ax, segment_table, stats_table, segment_length=100000):

    index_cols = ['chromosome', 'segment_start', 'sample_id']

    minor = segment_table.set_index(index_cols)['minor_raw']\
                         .unstack()\
                         .fillna(1.0)
    
    loh = 1.0 - minor
    loh = np.maximum(loh, 0.0)

    plot_cn_matrix(ax, loh, segment_length=segment_length)


def plot_total(ax, segment_table, stats_table, segment_length=100000):

    index_cols = ['chromosome', 'segment_start', 'sample_id']

    segment_matrix = segment_table.set_index(index_cols)[['major_raw', 'minor_raw']].unstack()

    total = segment_matrix['major_raw'] + segment_matrix['minor_raw']

    ploidy = stats_table.set_index('sample_id')['ploidy']

    for sample_id in total.keys():
        total[sample_id] /= ploidy[sample_id]

    total = total.fillna(1.0)
    
    plot_cn_matrix(ax, total, segment_length=segment_length, norm_max=2.0)


def plot_subclonal(ax, segment_table, stats_table, segment_length=100000):

    index_cols = ['chromosome', 'segment_start', 'sample_id']

    subclonal = segment_table.set_index(index_cols)['subclonal']\
                             .unstack()\
                             .fillna(0.0)

    plot_cn_matrix(ax, subclonal, segment_length=segment_length)


def plot_cn_matrix(ax, matrix, segment_length=100000, norm_min=0.0, norm_max=1.0, cmap=None):
    
    if cmap is None:
        cmap = plt.get_cmap('Oranges')

    chromosome_index = [(chrom, idx) for idx, chrom in enumerate(wgs_analysis.refgenome.info.chromosomes)]
    chromosome_index = pd.DataFrame(chromosome_index, columns=['chromosome', 'chromosome_idx'])

    matrix = (
        matrix
        .reset_index()
        .merge(chromosome_index)
        .drop('chromosome', axis=1)
        .set_index(['chromosome_idx', 'start'])
        .sort_index()
    )

    ax.imshow(
        matrix.values.T, aspect='auto', 
        interpolation='nearest',
        cmap=cmap,
        norm=matplotlib.colors.Normalize(vmin=norm_min, vmax=norm_max, clip=True),
    )

    chrom_seg_cnt = (
        matrix.reset_index()
        .groupby('chromosome_idx')
        .size()
        .sort_index()
    )

    chrom_seg_end = chrom_seg_cnt.cumsum()
    chrom_seg_mid = chrom_seg_end - chrom_seg_cnt / 2.0

    ax.set_xlim((-0.5, len(matrix.index)))
    ax.set_xticks([0] + list(chrom_seg_end.values))
    ax.set_xticklabels([])
    ax.set_yticks(xrange(len(matrix.columns.values)))
    ax.set_yticklabels(matrix.columns.values)
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(chrom_seg_mid))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(wgs_analysis.refgenome.info.chromosomes))
    ax.xaxis.grid(True, which='major', color='white', linestyle='-', linewidth=2)
    ax.xaxis.grid(False, which='minor')
    ax.yaxis.grid(False, which='major')
    ax.yaxis.grid(False, which='minor')

