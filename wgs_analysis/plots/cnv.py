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


def plot_cnv_segments(ax, cnv, major_col='major_raw', minor_col='minor_raw'):
    """ Plot raw major/minor copy number as line plots

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): cnv table
        major_col (str): name of major copies column
        minor_col (str): name of minor copies column

    Plot major and minor copy number as line plots.  The columns 'start' and 'end'
    are expected and should be adjusted for full genome plots.  Values from the
    'major_col' and 'minor_col' columns are plotted.

    """ 

    segment_color_major = plt.get_cmap('RdBu')(0.1)
    segment_color_minor = plt.get_cmap('RdBu')(0.9)

    quad_color_major = colorConverter.to_rgba(segment_color_major, alpha=0.5)
    quad_color_minor = colorConverter.to_rgba(segment_color_minor, alpha=0.5)

    cnv = cnv.sort('start')

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

    major_segments = create_segments(cnv, major_col)
    minor_segments = create_segments(cnv, minor_col)
    ax.add_collection(matplotlib.collections.LineCollection(major_segments, colors=segment_color_major, lw=1))
    ax.add_collection(matplotlib.collections.LineCollection(minor_segments, colors=segment_color_minor, lw=1))

    major_connectors = create_connectors(cnv, major_col)
    minor_connectors = create_connectors(cnv, minor_col)
    ax.add_collection(matplotlib.collections.LineCollection(major_connectors, colors=segment_color_major, lw=1))
    ax.add_collection(matplotlib.collections.LineCollection(minor_connectors, colors=segment_color_minor, lw=1))

    major_quads = create_quads(cnv, major_col)
    minor_quads = create_quads(cnv, minor_col)
    ax.add_collection(matplotlib.collections.PolyCollection(major_quads, facecolors=quad_color_major, edgecolors=quad_color_major, lw=0))
    ax.add_collection(matplotlib.collections.PolyCollection(minor_quads, facecolors=quad_color_minor, edgecolors=quad_color_minor, lw=0))


def plot_cnv_genome(ax, cnv, sample_id, maxcopies=4, minlength=1000):
    """
    Plot major/minor copy number across the genome

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): `cnv_site` table
        sample_id (str): site to plot
        maxcopies (int): maximum number of copies for setting y limits
        minlength (int): minimum length of segments to be drawn

    """

    cnv = cnv[cnv['sample_id'] == sample_id].copy()

    cnv = cnv[['chromosome', 'start', 'end', 'length', 'major_raw', 'minor_raw']]

    cnv = cnv[cnv['length'] >= minlength]
    
    cnv = cnv[cnv['chromosome'].isin(wgs_analysis.refgenome.info.chromosomes)]

    cnv.set_index('chromosome', inplace=True)
    cnv['chromosome_start'] = wgs_analysis.refgenome.info.chromosome_start
    cnv.reset_index(inplace=True)

    cnv['start'] = cnv['start'] + cnv['chromosome_start']
    cnv['end'] = cnv['end'] + cnv['chromosome_start']

    plot_cnv_segments(ax, cnv)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylim((-0.05*maxcopies, maxcopies+.6))
    ax.set_xlim((-0.5, wgs_analysis.refgenome.info.chromosome_end.max()))
    ax.set_xlabel('chromosome')
    ax.set_xticks([0] + list(wgs_analysis.refgenome.info.chromosome_end.values))
    ax.set_xticklabels([])
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.set_ylabel(sample_id)
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(wgs_analysis.refgenome.info.chromosome_mid))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(wgs_analysis.refgenome.info.chromosomes))


def plot_cnv_chromosome(ax, cnv, sample_id, chromosome, start=None, end=None, maxcopies=4, minlength=1000, fontsize=None):
    """
    Plot major/minor copy number across a chromosome

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): `cnv_site` table
        sample_id (str): site to plot
        chromosome (str): chromosome to plot
        start (int): start of range in chromosome to plot
        end (int): start of range in chromosome to plot
        maxcopies(float): maximum number of copies for setting y limits
        minlength(int): minimum length of segments to be drawn

    """

    cnv = cnv[cnv['sample_id'] == sample_id]
    
    cnv = cnv.loc[(cnv['chromosome'] == chromosome)]
    cnv = cnv.loc[(cnv['length'] >= minlength)]

    cnv['genomic_length'] = cnv['end'] - cnv['start']
    cnv['length_fraction'] = cnv['length'].astype(float) / cnv['genomic_length'].astype(float)
    cnv = cnv.loc[(cnv['length_fraction'] >= 0.5)]
    
    if start is not None:
        cnv = cnv.loc[(cnv['end'] > start)]
    
    if end is not None:
        cnv = cnv.loc[(cnv['start'] < end)]
    
    plot_cnv_segments(ax, cnv)
    
    if maxcopies is None:
        maxcopies = int(np.ceil(cnv['major_raw'].max()))

    ax.set_xlim((start, end))
    ax.set_ylim((-0.05*maxcopies, maxcopies+.6))
    
    ax.set_ylabel(helpers.plot_ids[sample_id])

    x_ticks_mb = ['{0:.3g}M'.format(x/1000000.) for x in ax.get_xticks()]
    
    ax.set_xticklabels(x_ticks_mb, fontsize=fontsize)
    ax.set_xlabel('Chromosome {0}'.format(chromosome))
    ax.set_yticks(xrange(0, int(maxcopies+1.), int(math.ceil(maxcopies/6.))))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plots.utils.trim_spines_to_ticks(ax)


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
    seg_idx_1, reseg_idx_1 = wgs_analysis.algorithms.merge.interval_position_overlap(
        cnv[['start', 'end']].values,
        uniform_segments['start'].values)
    reseg_1 = pd.DataFrame({'seg_idx': seg_idx_1, 'reseg_idx': reseg_idx_1})

    # Find ends of the resegmentation that fall within cnv segments
    seg_idx_2, reseg_idx_2 = wgs_analysis.algorithms.merge.interval_position_overlap(
        cnv[['start', 'end']].values,
        uniform_segments['end'].values)
    reseg_2 = pd.DataFrame({'seg_idx': seg_idx_2, 'reseg_idx': reseg_idx_2})

    # Find starts of the cnv segments that fall within the resegmentation
    reseg_idx_3, seg_idx_3 = wgs_analysis.algorithms.merge.interval_position_overlap(
        uniform_segments[['start', 'end']].values,
        cnv['start'].values)
    reseg_3 = pd.DataFrame({'seg_idx': seg_idx_3, 'reseg_idx': reseg_idx_3})

    reseg = pd.concat([reseg_1, reseg_2, reseg_3]).drop_duplicates()
    
    cnv = cnv.merge(reseg, on='seg_idx')
    cnv = cnv.merge(uniform_segments, on='reseg_idx', suffixes=('', '_reseg'))

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


def uniform_segment_copies(cnv, columns, segment_length=100000):
    """
    Create a table of uniformly segmented data from arbitrarily segment data

    Args:
        cnv (pandas.DataFrame): segment data
        columns (list): columns to resegment
        segment_length (int): uniform segment length

    Returns:
        pandas.DataFrame: resegmented table

    The cnv table should have sample_id, chrom, start and end columns in addition to
    the columns for which resegmentation is requested.  Returns a resegmented table
    with sample_id, chrom, segment_start, segment_end columns, in addition to the 
    requested columns calculated as length weighted averages of the original values.
    """

    cnv_reseg = uniform_resegment_genome(cnv, segment_length=100000)

    cnv_reseg = cnv_reseg.merge(cnv, on=['chromosome', 'start', 'end'])

    cnv_reseg['segment_start'] = cnv_reseg['start_reseg'] / segment_length
    cnv_reseg['segment_start'] = cnv_reseg['segment_start'].astype(int) * segment_length + 1

    cnv_reseg.set_index(['sample_id', 'chromosome', 'segment_start'], inplace=True)

    # Average requested columns weighted by length of segment
    # exclude null values in calculation
    for column in columns:

        # Length of resegmented segments
        cnv_reseg['length_reseg'] = cnv_reseg['end_reseg'] - cnv_reseg['start_reseg'] + 1

        # Mask segments with null values from normalization calculation
        cnv_reseg['length_reseg'] *= (cnv_reseg[column].notnull() * 1)

        # Normalize by total length of resegmented segments
        cnv_reseg['length_total_reseg'] = cnv_reseg.groupby(level=[0, 1, 2])['length_reseg'].sum()
        cnv_reseg['weight_reseg'] = cnv_reseg['length_reseg'].astype(float) / \
                                    cnv_reseg['length_total_reseg'].astype(float)
        cnv_reseg[column] *= cnv_reseg['weight_reseg']

        # Mask segments with null values from summation
        cnv_reseg[column] = cnv_reseg[column].fillna(0.0)
    
    cnv_reseg = cnv_reseg.groupby(level=[0, 1, 2])[columns].sum()

    cnv_reseg.reset_index(inplace=True)

    # Ensure the segments are consistent regardless of the cnv data
    seg_full = (
        create_uniform_segments_genome(segment_length)
        .rename(columns={'start':'segment_start'})
        .set_index(['chromosome', 'segment_start']))

    cnv_reseg = cnv_reseg.set_index(['chromosome', 'segment_start', 'sample_id'])\
                         .unstack()\
                         .reindex(seg_full.index)\
                         .stack(dropna=False)\
                         .reset_index()

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

