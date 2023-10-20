'''
# SNV plotting

'''

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn

import wgs_analysis.plots as plots
import wgs_analysis.plots.colors
import wgs_analysis.refgenome as refgenome


def snv_adjacent_density_plot(ax, snvs, color_by_chromosome=True, alpha=0.5, s=5, lw=0, **kwargs):

    snvs = snvs.drop_duplicates(['chrom', 'coord'])

    snvs = snvs.loc[(snvs['chrom'].isin(refgenome.info.chromosomes))]

    snvs.set_index('chrom', inplace=True)
    snvs['chromosome_start'] = refgenome.info.chromosome_start
    snvs['chromosome_color'] = pd.Series(plots.colors.create_chromosome_color_map())
    snvs.reset_index(inplace=True)

    snvs['plot_coord'] = snvs['coord'] + snvs['chromosome_start']

    assert not snvs['chromosome_color'].isnull().any()

    facecolors = None
    edgecolors = None
    if color_by_chromosome:
        facecolors = list(snvs['chromosome_color'])
        edgecolors = list(snvs['chromosome_color'])

    ax.scatter(snvs['plot_coord'], snvs['adjacent_density'], 
                facecolors=facecolors,
                edgecolors=edgecolors,
                alpha=alpha, s=s, lw=lw, **kwargs)

    ax.set_xlim(min(snvs['plot_coord']), max(snvs['plot_coord']))
    ax.set_xticks(refgenome.info.chromosome_mid, minor=False)
    ax.set_xticklabels(refgenome.info.chromosomes, minor=False)
    ax.set_xticks(refgenome.info.chromosome_end, minor=True)
    ax.set_xticklabels([], minor=True)
    ax.grid(False, which='major')

    ax.set_ylim(-0.0001, 1.1 * snvs['adjacent_density'].fillna(0.).max())
    ax.set_ylabel('snv density')


def snv_adjacent_distance_plot(ax, snvs, color_by_chromosome=True, alpha=0.5, s=5, lw=0, **kwargs):

    snvs = snvs.drop_duplicates(['chrom', 'coord'])

    snvs = snvs.loc[(snvs['chrom'].isin(refgenome.info.chromosomes))]

    snvs.set_index('chrom', inplace=True)
    snvs['chromosome_start'] = refgenome.info.chromosome_start
    snvs['chromosome_color'] = pd.Series(plots.colors.create_chromosome_color_map())
    snvs.reset_index(inplace=True)

    snvs['plot_coord'] = snvs['coord'] + snvs['chromosome_start']

    assert not snvs['chromosome_color'].isnull().any()

    snvs['adjacent_distance_log'] = snvs['adjacent_distance'].apply(np.log10)

    facecolors = None
    edgecolors = None
    if color_by_chromosome:
        facecolors = list(snvs['chromosome_color'])
        edgecolors = list(snvs['chromosome_color'])

    ax.scatter(
        snvs['plot_coord'],
        snvs['adjacent_distance_log'], 
        facecolors=facecolors,
        edgecolors=edgecolors,
        alpha=alpha, s=s, lw=lw, **kwargs)

    ax.set_xlabel('chromosome')
    ax.set_xlim(min(snvs['plot_coord']), max(snvs['plot_coord']))
    ax.set_xticks([0] + list(wgs_analysis.refgenome.info.chromosome_end.values))
    ax.set_xticklabels([])
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(wgs_analysis.refgenome.info.chromosome_mid))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(wgs_analysis.refgenome.info.chromosomes))

    ax.set_ylim(-0.0001, 1.1 * snvs['adjacent_distance_log'].fillna(0.).max())
    ax.set_ylabel('Distance between mutations (log10)')

    seaborn.despine(trim=True)


def snv_genome_vaf_plot(ax, snvs):
    snvs = snvs.drop_duplicates(['chrom', 'coord'])

    snvs = snvs.loc[(snvs['chrom'].isin(refgenome.info.chromosomes))]

    snvs.set_index('chrom', inplace=True)
    snvs['chromosome_start'] = refgenome.info.chromosome_start
    snvs['chromosome_color'] = pd.Series(plots.colors.create_chromosome_color_map())
    snvs.reset_index(inplace=True)

    snvs['plot_coord'] = snvs['coord'] + snvs['chromosome_start']

    assert not snvs['chromosome_color'].isnull().any()

    ax.scatter(
        snvs['plot_coord'],
        snvs['vaf'], 
        facecolors=list(snvs['chromosome_color']),
        edgecolors=list(snvs['chromosome_color']),
        alpha=0.5, s=5, lw=0)

    ax.set_xlabel('chromosome')
    ax.set_xlim(min(snvs['plot_coord']), max(snvs['plot_coord']))
    ax.set_xticks([0] + list(wgs_analysis.refgenome.info.chromosome_end.values))
    ax.set_xticklabels([])
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(wgs_analysis.refgenome.info.chromosome_mid))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(wgs_analysis.refgenome.info.chromosomes))

    ax.set_ylim(-0.01, 1.01)
    ax.set_ylabel('VAF')

    seaborn.despine(trim=True)


def snv_count_plot(ax, snvs, bin_size=10000000, full_genome=True):
    snvs = snvs.loc[(snvs['chrom'].isin(refgenome.info.chromosomes))].copy()

    snvs.set_index('chrom', inplace=True)
    snvs['chromosome_start'] = refgenome.info.chromosome_start
    snvs.reset_index(inplace=True)

    snvs['plot_position'] = snvs['coord'] + snvs['chromosome_start']
    snvs['plot_bin'] = snvs['plot_position'] / bin_size
    snvs['plot_bin'] = snvs['plot_bin'].astype(int)

    if full_genome:
        plot_bin_starts = np.arange(0, refgenome.info.chromosome_end.max(), bin_size)
    else:
        min_x, max_x = snvs['plot_position'].min(), snvs['plot_position'].max()
        plot_bin_starts = np.arange(min_x, max_x, bin_size)
    plot_bins = (plot_bin_starts / bin_size).astype(int)

    snvs = snvs[['chrom', 'coord', 'ref', 'alt', 'plot_bin']].drop_duplicates()

    count_table = snvs.groupby(['plot_bin']).size()
    count_table = count_table.reindex(plot_bins).fillna(0)

    ax.bar(plot_bin_starts, count_table.values, width=bin_size)

    if full_genome:
        ax.set_xlim((-0.5, refgenome.info.chromosome_end.max()))
    else:
        ax.set_xlim((min_x, max_x))
    ax.set_xlabel('chrom')
    ax.set_xticks([0] + list(refgenome.info.chromosome_end.values))
    ax.set_xticklabels([])
    ax.set_ylabel('count')
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(refgenome.info.chromosome_mid))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(refgenome.info.chromosomes))
    ax.xaxis.grid(False, which="minor")
