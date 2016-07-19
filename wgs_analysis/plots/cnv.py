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
import seaborn

import ith_project.analysis.plots as plots
import ith_project.analysis.plots.colors
import ith_project.analysis.plots.positions
import ith_project.analysis.plots.utils
import ith_project.analysis.algorithms as algorithms
import ith_project.analysis.algorithms.cnv



def plot_segments(ax, cnv, value_col, color):
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

    cnv = cnv.sort('start')

    def plot_segment(ax, row, field, color):
        ax.plot([row['start'], row['end']], [row[field]]*2, color=color, lw=1)
    
    def plot_connectors(ax, row, next_row, field, color):
        mid = (row[field] + next_row[field]) / 2.0
        ax.plot([row['end'], row['end']], [row[field], mid], color=color, lw=1)
        ax.plot([next_row['start'], next_row['start']], [mid, next_row[field]], color=color, lw=1)
    
    for (idx, row), (next_idx, next_row) in itertools.izip_longest(cnv.iterrows(), cnv.iloc[1:].iterrows(), fillvalue=(None, None)):
        plot_segment(ax, row, value_col, color)
        if next_row is not None:
            plot_connectors(ax, row, next_row, value_col, color)


def plot_cnv_segments(ax, cnv):
    """
    Plot raw major/minor copy number as line plots

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): cnv table

    Plot major and minor copy number as line plots.  The columns 'start' and 'end'
    are expected and should be adjusted for full genome plots.  Values from  the
    'major_raw' and 'minor_raw' columns are plotted.

    """ 

    color_major = plt.get_cmap('RdBu')(0.1)
    color_minor = plt.get_cmap('RdBu')(0.9)

    cnv = cnv.sort('start')

    def plot_segment(ax, row, field, color):
        ax.plot([row['start'], row['end']], [row[field]]*2, color=color, lw=1)
    
    def plot_connectors(ax, row, next_row, field, color):
        mid = (row[field] + next_row[field]) / 2.0
        ax.plot([row['end'], row['end']], [row[field], mid], color=color, lw=1)
        ax.plot([next_row['start'], next_row['start']], [mid, next_row[field]], color=color, lw=1)
    
    for (idx, row), (next_idx, next_row) in itertools.izip_longest(cnv.iterrows(), cnv.iloc[1:].iterrows(), fillvalue=(None, None)):
        plot_segment(ax, row, 'major_raw', color_major)
        plot_segment(ax, row, 'minor_raw', color_minor)
        if next_row is not None:
            plot_connectors(ax, row, next_row, 'major_raw', color_major)
            plot_connectors(ax, row, next_row, 'minor_raw', color_minor)


def plot_cnv_genome(ax, cnv, site_id, maxcopies=4, minlength=1000):
    """
    Plot major/minor copy number across the genome

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): `cnv_site` table
        site_id (str): site to plot
        maxcopies (int): maximum number of copies for setting y limits
        minlength (int): minimum length of segments to be drawn

    """

    cnv = cnv[cnv['site_id'] == site_id].copy()

    cnv = cnv[['chrom', 'start', 'end', 'length', 'major_raw', 'minor_raw']]

    cnv = cnv[cnv['length'] >= minlength]
    
    cnv = cnv[cnv['chrom'].isin(plots.utils.chromosomes)]

    cnv.set_index('chrom', inplace=True)
    cnv['chromosome_start'] = plots.utils.chromosome_start
    cnv.reset_index(inplace=True)

    cnv['start'] = cnv['start'] + cnv['chromosome_start']
    cnv['end'] = cnv['end'] + cnv['chromosome_start']

    plot_cnv_segments(ax, cnv)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylim((-0.05*maxcopies, maxcopies+.6))
    ax.set_xlim((-0.5, plots.utils.chromosome_end.max()))
    ax.set_xlabel('chrom')
    ax.set_xticks([0] + list(plots.utils.chromosome_end.values))
    ax.set_xticklabels([])
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.set_ylabel(helpers.plot_ids[site_id])
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(plots.utils.chromosome_mid))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(plots.utils.chromosomes))


def plot_cnv_chromosome(ax, cnv, site_id, chromosome, start=None, end=None, maxcopies=4, minlength=1000, fontsize=None):
    """
    Plot major/minor copy number across a chromosome

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): `cnv_site` table
        site_id (str): site to plot
        chromosome (str): chromosome to plot
        start (int): start of range in chromosome to plot
        end (int): start of range in chromosome to plot
        maxcopies(float): maximum number of copies for setting y limits
        minlength(int): minimum length of segments to be drawn

    """

    cnv = cnv[cnv['site_id'] == site_id]
    
    cnv = cnv.loc[(cnv['chrom'] == chromosome)]
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
    
    ax.set_ylabel(helpers.plot_ids[site_id])

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

    for fig_idx, site_id in enumerate(site_ids):
        
        ax = plots.utils.setup_axes(plt.subplot(num_sites, 1, fig_idx+1))
        
        plots.cnv.plot_cnv_chromosome(ax, cnv, site_id, chromosome, start=start, end=end, maxcopies=maxcopies)

        if fig_idx != num_sites - 1:
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_ticks_position('none')

        site_axes[site_id] = ax
        
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

    for fig_idx, site_id in enumerate(site_ids):
        
        site_brks = brks[brks['site_id'] == site_id]
        
        ax = plots.utils.setup_axes(plt.subplot(num_sites, 1, fig_idx+1))
        
        plots.cnv.plot_cnv_chromosome(ax, cnv, site_id, chromosome, start=start, end=end, maxcopies=maxcopies)
        plots.positions.plot_breakends(ax, brks, site_id, chromosome, start=start, end=end)

        if fig_idx != num_sites - 1:
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_ticks_position('none')

        site_axes[site_id] = ax
        
    plt.tight_layout()
    
    return site_axes


def plot_cnv_scatter(ax, cnv, site, chromosome_color, point_alpha=1.0, line_alpha=1.0):

    cnv = cnv[cnv['site_id'] == site]

    cnv = cnv[cnv['length'] >= 10000]
    
    cnv = cnv.set_index(['chrom', 'start', 'end'])

    cs = [chromosome_color[c[0]] for c in cnv.index.values]
    sz = cnv['length'] / 100000.

    points = ax.scatter(cnv['major_raw'], cnv['minor_raw'], s=sz, facecolor=cs, edgecolor=cs, linewidth=0.0, zorder=2, alpha=point_alpha)

    ax.set_xlim((-0.5, cnv['major_raw'].quantile(0.95) + 0.5))
    ax.set_ylim((-0.5, cnv['minor_raw'].quantile(0.95) + 0.5))


def plot_cnv_scatter_pairwise(ax, cnv, site_a, site_b, chromosome_color, point_alpha=1.0, line_alpha=1.0):

    cnv = cnv.set_index(['chrom', 'start', 'end', 'site_id']).unstack()
    cnv = cnv.fillna(0.0)

    cs = [chromosome_color[c[0]] for c in cnv.index.values]
    sz = (cnv['length'][site_a] + cnv['length'][site_b])/500000.

    for x1, y1, x2, y2, c, s in zip(cnv['major_raw'][site_a], cnv['minor_raw'][site_a], cnv['major_raw'][site_b], cnv['minor_raw'][site_b], cs, sz):
        lines = plt.plot([x1, x2], [y1, y2], alpha=line_alpha, zorder=1, color=c, lw=1.0)

    points1 = ax.scatter(cnv['major_raw'][site_a], cnv['minor_raw'][site_a], s=sz, facecolor=cs, edgecolor=cs, linewidth=0.0, zorder=2, alpha=point_alpha)
    points2 = ax.scatter(cnv['major_raw'][site_b], cnv['minor_raw'][site_b], s=sz, facecolor=cs, edgecolor=cs, linewidth=0.0, zorder=2, alpha=point_alpha)


def plot_patient_cnv(cnv):

    sites = cnv['site_id'].unique()

    plots.utils.setup_plot()
    fig = plt.figure(figsize=(12,5*len(sites)))

    chromosome_color = plots.colors.create_chromosome_color_map()

    for idx, site_id in enumerate(sites):
        ax = plots.utils.setup_axes(plt.subplot(len(sites), 1, idx+1))
        plot_cnv_scatter(ax, cnv, site_id, chromosome_color)
        if idx == len(sites) - 1:
            ax.set_xlabel('major copy number')
            ax.set_ylabel('minor copy number')
        ax.set_title(helpers.plot_ids[site_id] + ' copies')

    fig.legend([plt.Circle((0, 0), radius=1, color=color) for chrom, color in chromosome_color.items()], chromosome_color.keys(), loc="lower right")

    plt.tight_layout()

    return fig


def plot_cnv_pairwise(ax, cnv, site_a, site_b, chromosome_color, highlighted_chromosomes=None):

    cnv = cnv[(cnv['site_id'] == site_a) | (cnv['site_id'] == site_b)]
    
    cnv = algorithms.cnv.cluster_copies(cnv)
    
    cnv = cnv[cnv['length'] >= 5000000]
    
    if highlighted_chromosomes is None:
        plot_cnv_scatter_pairwise(ax, cnv, site_a, site_b, chromosome_color, 1.0, 0.5)
    else:
        plot_cnv_scatter_pairwise(ax, cnv[~(cnv['chrom'].isin(highlighted_chromosomes))], site_a, site_b, chromosome_color, 0.1, 0.1)
        plot_cnv_scatter_pairwise(ax, cnv[cnv['chrom'].isin(highlighted_chromosomes)], site_a, site_b, chromosome_color, 1.0, 1.0)

    ax.grid(True)


def plot_patient_cnv_pairwise(fig, cnv, highlighted_chromosomes=None):

    pairs = list(itertools.combinations(cnv['site_id'].unique(), 2))

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


def create_uniform_segments(segment_length):
    """
    Create a table of uniform segments with a given segment length

    Args:
        segment_length (int): uniform segment length

    Returns:
        pandas.DataFrame: table of uniform segments

    The returned table will have columns chrom, start, end.  Segments are created according
    to chromosome lengths.
    """

    num_segments = (plots.utils.chromosome_lengths.astype(float) / segment_length).apply(np.ceil)

    chroms = np.concatenate([np.repeat(c, n) for c, n in num_segments.iteritems()])
    starts = np.concatenate([np.arange(0, m*segment_length, segment_length) for c, m in num_segments.iteritems()])

    chroms = chroms.astype(str)
    starts = starts.astype(int) + 1

    segments = pd.DataFrame({'chrom':chroms, 'start':starts})
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

    The cnv table should have columns chrom, start, end.  Returns a resegmented
    table with columns chrom, start, end, start_reseg, end_reseg.  Original 
    rows will be represented multiple times if they are split by uniform segments,
    and can be identified by having the same chrom, start, end.  The start_reseg
    and end_reseg columns represent the subsegment resulting from the split.
    """

    # First segment cannot start at coordinate less than 1
    assert cnv['start'].min() >= 1

    # Uniquely index segments
    cnv = cnv[['chrom', 'start', 'end']].drop_duplicates()
    cnv['idx'] = xrange(len(cnv.index))

    # Set of start coordinates
    cn_starts = cnv[['idx', 'chrom', 'start']]

    # Create a table of fill segments
    cn_fill = cnv[['chrom', 'end']].rename(columns={'end':'start'})

    # Annotate last segment of each chromosome
    cn_fill.set_index('chrom', inplace=True)
    cn_fill['max_start'] = cn_fill.groupby(level=0)['start'].max()
    cn_fill.reset_index(inplace=True)

    # Last segment of each chromosome is wrapped around to be used as first segment in fill
    cn_fill.loc[cn_fill['start'] == cn_fill['max_start'], 'start'] = 1
    cn_fill = cn_fill.drop('max_start', axis=1)
    cn_fill = cn_fill.drop_duplicates()

    # Add fill starts to original segment starts
    cn_starts = cn_starts.merge(cn_fill, on=['chrom', 'start'], how='outer')
    cn_starts['idx'] = cn_starts['idx'].fillna(-1).astype(int)

    # Create table of uniform segment starts
    uniform_starts = create_uniform_segments(segment_length)[['chrom', 'start']]

    # Create a union set of segment start points
    union_starts = cn_starts.merge(uniform_starts, on=['chrom', 'start'], how='outer')

    # Intermediate start coordinates with NAN index are uniform segment
    # boundaries for uniform segments that intersect original segments.
    # Set the idx for these to that of the previous start to mark them
    # as a continuation of the previous original segment that has been
    # cut somewhere in the middle.
    union_starts = union_starts.sort(['chrom', 'start'])
    union_starts['idx'] = union_starts['idx'].fillna(method='ffill').astype(int)
    union_starts = union_starts[union_starts['idx'] != -1]

    # Merge original segment information and select the end as the minimum
    # of the original segment or the uniform segment for segments that have
    # been cut
    union_segments = union_starts.merge(cnv[['idx', 'end']], on='idx')
    union_segments['end'] = np.minimum(union_segments['end'],
                                       union_segments['start'] + segment_length - 1)

    # Final segments will have original start and end, in addition to a
    # resegmented start_1 and end_1
    union_segments = union_segments.merge(cnv[['idx', 'start', 'end']], on='idx', suffixes=('_reseg', ''))

    union_segments = union_segments.drop('idx', axis=1)

    return union_segments


def uniform_segment_copies(cnv, columns, segment_length=100000):
    """
    Create a table of uniformly segmented data from arbitrarily segment data

    Args:
        cnv (pandas.DataFrame): segment data
        columns (list): columns to resegment
        segment_length (int): uniform segment length

    Returns:
        pandas.DataFrame: resegmented table

    The cnv table should have site_id, chrom, start and end columns in addition to
    the columns for which resegmentation is requested.  Returns a resegmented table
    with site_id, chrom, segment_start, segment_end columns, in addition to the 
    requested columns calculated as length weighted averages of the original values.
    """

    cnv_reseg = uniform_resegment(cnv, segment_length=100000)

    cnv_reseg = cnv_reseg.merge(cnv, on=['chrom', 'start', 'end'])

    cnv_reseg['segment_start'] = cnv_reseg['start_reseg'] / segment_length
    cnv_reseg['segment_start'] = cnv_reseg['segment_start'].astype(int) * segment_length + 1

    cnv_reseg.set_index(['site_id', 'chrom', 'segment_start'], inplace=True)

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
    seg_full = create_uniform_segments(segment_length).rename(columns={'start':'segment_start'})\
                                                      .set_index(['chrom', 'segment_start'])

    cnv_reseg = cnv_reseg.set_index(['chrom', 'segment_start', 'site_id'])\
                         .unstack()\
                         .reindex(seg_full.index)\
                         .stack(dropna=False)\
                         .reset_index()

    cnv_reseg['segment_end'] = cnv_reseg['segment_start'] + segment_length - 1

    return cnv_reseg


def plot_loh(ax, segment_table, stats_table, segment_length=100000):

    index_cols = ['chrom', 'segment_start', 'site_id']

    minor = segment_table.set_index(index_cols)['minor_raw']\
                         .unstack()\
                         .fillna(1.0)
    
    loh = 1.0 - minor
    loh = np.maximum(loh, 0.0)

    plot_cn_matrix(ax, loh, segment_length=segment_length)


def plot_total(ax, segment_table, stats_table, segment_length=100000):

    index_cols = ['chrom', 'segment_start', 'site_id']

    segment_matrix = segment_table.set_index(index_cols)[['major_raw', 'minor_raw']].unstack()

    total = segment_matrix['major_raw'] + segment_matrix['minor_raw']

    ploidy = stats_table.set_index('site_id')['ploidy']

    for site_id in total.keys():
        total[site_id] /= ploidy[site_id]

    total = total.fillna(1.0)
    
    plot_cn_matrix(ax, total, segment_length=segment_length, norm_max=2.0)


def plot_subclonal(ax, segment_table, stats_table, segment_length=100000):

    index_cols = ['chrom', 'segment_start', 'site_id']

    subclonal = segment_table.set_index(index_cols)['subclonal']\
                             .unstack()\
                             .fillna(0.0)

    plot_cn_matrix(ax, subclonal, segment_length=segment_length)


def plot_cn_matrix(ax, matrix, segment_length=100000, norm_min=0.0, norm_max=1.0):

    chromosome_index = [(chrom, idx) for idx, chrom in enumerate(plots.utils.chromosomes)]
    chromosome_index = pd.DataFrame(chromosome_index, columns=['chrom', 'chrom_idx'])

    matrix = matrix.reset_index()\
                   .merge(chromosome_index)\
                   .drop('chrom', axis=1)\
                   .set_index(['chrom_idx', 'segment_start'])\
                   .sort_index()\

    ax.imshow(matrix.values.T, aspect='auto', 
              interpolation='nearest',
              cmap=plt.get_cmap('Oranges'),
              norm=matplotlib.colors.Normalize(vmin=norm_min, vmax=norm_max, clip=True))

    chrom_seg_cnt = matrix.reset_index()\
                          .groupby('chrom_idx')\
                          .size()\
                          .sort_index()

    chrom_seg_end = chrom_seg_cnt.cumsum()
    chrom_seg_mid = chrom_seg_end - chrom_seg_cnt / 2.0

    ax.set_xlim((-0.5, len(matrix.index)))
    ax.set_xticks([0] + list(chrom_seg_end.values))
    ax.set_xticklabels([])
    ax.set_yticks(xrange(len(matrix.columns.values)))
    ax.set_yticklabels([helpers.plot_ids[a] for a in matrix.columns.values])
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(chrom_seg_mid))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(plots.utils.chromosomes))
    ax.xaxis.grid(True, which='major', color='white', linestyle='-', linewidth=2)
    ax.xaxis.grid(False, which='minor')
    ax.yaxis.grid(False, which='major')
    ax.yaxis.grid(False, which='minor')

