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


def snv_adjacent_density_plot(ax, snvs):

    snvs = snvs.drop_duplicates(['chrom', 'coord'])

    snvs = snvs.loc[(snvs['chrom'].isin(refgenome.info.chromosomes))]

    snvs.set_index('chrom', inplace=True)
    snvs['chromosome_start'] = refgenome.info.chromosome_start
    snvs['chromosome_color'] = pd.Series(plots.colors.create_chromosome_color_map())
    snvs.reset_index(inplace=True)

    snvs['plot_coord'] = snvs['coord'] + snvs['chromosome_start']

    assert not snvs['chromosome_color'].isnull().any()

    ax.scatter(snvs['plot_coord'], snvs['adjacent_density'], 
                facecolors=list(snvs['chromosome_color']),
                edgecolors=list(snvs['chromosome_color']),
                alpha=0.5, s=5, lw=0)

    ax.set_xlim(min(snvs['plot_coord']), max(snvs['plot_coord']))
    ax.set_xticks(refgenome.info.chromosome_mid, minor=False)
    ax.set_xticklabels(refgenome.info.chromosomes, minor=False)
    ax.set_xticks(refgenome.info.chromosome_end, minor=True)
    ax.set_xticklabels([], minor=True)
    ax.grid(False, which='major')

    ax.set_ylim(-0.0001, 1.1 * snvs['adjacent_density'].max())
    ax.set_ylabel('snv density')

