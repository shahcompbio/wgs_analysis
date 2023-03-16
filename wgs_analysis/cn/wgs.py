import os
import sys
import gzip
import argparse
import logging
from collections import defaultdict 
from collections.abc import Iterable

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.font_manager as font_manager
import pandas as pd
import numpy as np
import pyranges as pr
from beartype import beartype
from fisher import pvalue 

import wgs_analysis.refgenome
import wgs_analysis.plots.cnv
import shahlabdata.wgs


@beartype
def get_include_wgs_samples(cohort:str) -> list:
    """ Return a list of WGS samples to include in the cohort
    """ 
    include_wgs_samples = []
    if cohort == "SPECTRUM":
        samples_path = '/juno/work/shah/users/chois7/tickets/cohort-cn-qc/include.SPECTRUM.txt'
        include_wgs_samples = [x.strip() for x in open(samples_path, 'r').readlines()]
    return include_wgs_samples

@beartype
def qc_cn_data_and_samples(cn_data:pd.DataFrame) -> None:
    """ Check input cn_data shape
    """ 
    loggong.debug(f"cn_data.shape: {cn_data.shape}")
    n_samples = cn_data.isabl_sample_id.unique().shape
    logging.debug(f"unique sample count: {n_samples}")

@beartype
def select_histotype_and_project(cn_data:pd.DataFrame, cohort:str) -> pd.DataFrame:
    """ Return subset of cn_data by histotype and cohort
    """
    if cohort == "Metacohort":
        labels = pd.read_csv('/juno/work/shah/users/chois7/spectrum/cnv_gistic/ov-tnbc_labels_fixed.tsv', sep='\t')
        labels = labels[labels['histotype'] == 'HGSC']
        labels = labels[labels['histotype'].notnull()]
        labels = labels[labels['cohort'] != 'APOLLO']
        cn_data = cn_data.merge(labels[['isabl_sample_id']].drop_duplicates())
    return cn_data

@beartype
def rebin(data:pd.DataFrame, bin_size:int, cols:list, chroms:list) -> pd.DataFrame:
    """ Merge/split data bins to given bin_size
    """
    genome_fai = '/work/shah/users/mcphera1/remixt/ref_data_hg19/Homo_sapiens.GRCh37.70.dna.chromosomes.fa.fai'
    chromsizes = pd.read_csv(
        genome_fai, sep='\t',
        dtype={'Chromosome': str},
        names=['Chromosome', 'End', 'A', 'B', 'C'],
        usecols=['Chromosome', 'End']).assign(Start=0)
    chromsizes = chromsizes[chromsizes['Chromosome'].isin(chroms)]

    chromsizes = pr.PyRanges(chromsizes)

    bins = pr.gf.tile_genome(chromsizes, bin_size)
    bins = bins.insert(pd.Series(range(len(bins)), name='bin'))

    data2 = pr.PyRanges(data.rename(columns={
        'chromosome': 'Chromosome',
        'start': 'Start',
        'end': 'End',
    }))

    intersect_1 = data2.intersect(bins)
    intersect_2 = bins.intersect(data2)

    intersect = pd.merge(
        intersect_1.as_df(), intersect_2.as_df(),
        on=['Chromosome', 'Start', 'End'], how='outer')

    intersect['length'] = intersect['End'] - intersect['Start'] + 1

    for col in cols:
        intersect[col] = intersect[col] * intersect['length'] / bin_size
    intersect = intersect.groupby('bin')[cols].sum().reset_index()
    intersect = intersect.merge(bins.as_df())
    intersect = intersect.rename(columns={
        'Chromosome': 'chromosome',
        'Start': 'start',
        'End': 'end',
    })

    return intersect[['chromosome', 'start', 'end'] + cols]

@beartype
def get_per_sample_data_and_ploidy(cn_data:pd.DataFrame) -> tuple:
    """ Separate cn_data to create table per sample and a ploidy series
    """
    chroms = wgs_analysis.refgenome.info.chromosomes[:-1]
    data = {}
    ploidy = {}

    for sample_id, sample_cn in cn_data.groupby('isabl_sample_id'):
        sample_cn['is_valid'] = (
            (sample_cn['length'] > 100000) &
            (sample_cn['minor_readcount'] > 100) &
            (sample_cn['length_ratio'] > 0.8))

        sample_cn['total_raw'] = sample_cn['minor_raw_e'] + sample_cn['major_raw_e']

        sample_cn = sample_cn[sample_cn['is_valid']]
        sample_cn = rebin(sample_cn, 500000, ['total_raw'], chroms)
        sample_cn = sample_cn.set_index(['chromosome', 'start', 'end'])

        data[sample_id] = sample_cn['total_raw']
        ploidy[sample_id] = sample_cn['total_raw'].mean()

    data = pd.DataFrame(data)
    ploidy = pd.Series(ploidy)

    return data, ploidy

@beartype
def get_mean_above_background(data:pd.DataFrame) -> pd.DataFrame:
    """ Calculate copy number above mean copy number of nearby regions
    """
    mean_copy = data.mean(axis=1).rename('mean').sort_index()

    background = []
    for shift in list(range(-5, 6)):
        if shift == 0:
            continue
        background.append(mean_copy.shift(shift))
    background = pd.DataFrame(background)

    mean_above_background = mean_copy - background.mean()
    mean_above_background = mean_above_background.rename('mean_above_background').reset_index()
    return mean_above_background

@beartype
def create_cn_change_table(data:pd.DataFrame, ploidy:pd.Series,
        mean_above_background:pd.DataFrame) -> pd.DataFrame:
    """ Return calculated CN change per bin
    """
    cn_change_table = {}
    sample_ids = data.columns
    log_change = np.log2(data.loc[:, sample_ids]) - np.log2(ploidy.loc[sample_ids])

    gain = (log_change >= 0.5).sum(axis=1).rename('gain').reset_index()
    loss = (log_change < -0.5).sum(axis=1).rename('loss').reset_index()

    cn_change = pd.merge(gain, loss, on=['chromosome', 'start', 'end'])
    cn_change = cn_change.merge(mean_above_background)

    cn_change = cn_change.merge(data.mean(axis=1).rename('mean_cn').reset_index())
    cn_change = cn_change.merge(data.var(axis=1).rename('var_cn').reset_index())

    cn_change.loc[cn_change['mean_above_background'] > 1, 'gain'] = 0
    cn_change.loc[cn_change['mean_above_background'] < -0.3, 'loss'] = 0

    # cn_change.loc[cn_change['mean_cn'] > 5, 'gain'] = 0
    # cn_change.loc[cn_change['var_cn'] > 10, 'gain'] = 0
    cn_change.loc[cn_change['mean_cn'] < 1.5, 'gain'] = 0

    # cn_change.loc[cn_change['mean_cn'] > 5, 'loss'] = 0
    # cn_change.loc[cn_change['var_cn'] > 10, 'loss'] = 0
    cn_change.loc[cn_change['mean_cn'] < 1.5, 'loss'] = 0

    cn_change['length'] = cn_change['end'] - cn_change['start']
    cn_change.loc[cn_change['length'] < 500000, ['gain', 'loss']] = 0
    cn_change.loc[cn_change['start'] < 2000000, ['gain', 'loss']] = 0

    cn_change_table = cn_change
    logging.debug(f'create_cn_change_table: sample_ids count: {len(sample_ids)}')
    return cn_change_table

@beartype
def get_chromosome_gap_band_data() -> pd.DataFrame:
    """ Parse gap.txt.gz [external] and return gap annotation dataframe
    """
    gap_path = '/rtsess01/juno/home/chois7/ondemand/spectrumanalysis/analysis/notebooks/bulk-dna/data/gap.txt.gz'
    gap_table = pd.read_table(gap_path, sep='\t', 
                              names=['tag1', 'chromosome', 'start', 'end', 
                                     'tag2', 'tag3', 'length', 'annotation', 'tag4'])
    gap_table['chromosome'] = gap_table['chromosome'].str.replace('chr', '')
    centromere_table = gap_table[gap_table['annotation'] == "centromere"]
    return centromere_table

@beartype
def get_chrom_size_table(chroms:list) -> dict:
    """ Return dict with key: chrom -> value: chrom size
    """
    genome_fai = '/juno/work/shah/users/mcphera1/remixt/ref_data_hg19/Homo_sapiens.GRCh37.70.dna.chromosomes.fa.fai'
    chrom_size_table = {}
    
    chromsizes = pd.read_csv(
        genome_fai, sep='\t',
        dtype={'Chromosome': str},
        names=['Chromosome', 'End', 'A', 'B', 'C'],
        usecols=['Chromosome', 'End']).assign(Start=0)

    chromsizes = chromsizes[chromsizes['Chromosome'].isin(chroms)]
    chrom_size_table = {x: int(chromsizes.loc[chromsizes['Chromosome']==x, 'End'])
                        for x in chroms} # int(chrom end size)
    return chrom_size_table


@beartype
def get_filled_table(cn_change:pd.DataFrame, chrom_size_table:dict, chroms:list, bin_size:int):
    """ Create a new complete table including centromere data
    this table does not omit genomic regions, omitted values become zeros
    """
    df = pd.DataFrame()
    for chrom in chroms:
        chrom_size = chrom_size_table[chrom]

        n_bins = int(chrom_size/bin_size)
        binned_chrom_size = n_bins * bin_size
        start_pos = range(0, binned_chrom_size, bin_size)
        end_pos = range(bin_size, binned_chrom_size + bin_size, bin_size)

        chrom_table = pd.DataFrame({
            'chromosome': [chrom] * n_bins,
            'start': start_pos,
            'end': end_pos,
        })

        df = pd.concat([df, chrom_table])

    df = df.reset_index(drop=True)
    part_cn_change = cn_change.loc[:, ['chromosome', 'start', 'gain', 'loss', 'norm_gain', 'norm_loss']]
    df = df.merge(part_cn_change, how='outer', on=['chromosome', 'start'], sort=True)
    df = df.fillna(0)
    
    return df

    
@beartype
def label_centromere_regions(df:pd.DataFrame, centromere_table:pd.DataFrame, 
        chroms:list, bin_size:int) -> pd.DataFrame:
    """ Annotate centromeric regions (omitting telomreres since their size is ~10000bp < 500000bp)
    """
    
    df['centromere'] = False
    
    for chrom in chroms:
        centromere_row = centromere_table[centromere_table['chromosome'] == chrom]
        start = int(centromere_row.start)
        end = int(centromere_row.end)
        start = int(start / bin_size) * bin_size
        end = int(end / bin_size) * bin_size

        chrom_table = df.loc[df['chromosome']==chrom, :]
        chrom_table.loc[(chrom_table['start'] >= start) & (chrom_table['end'] <= end), 'centromere'] = True
        df.loc[df['chromosome'] == chrom, 'centromere'] = chrom_table.centromere
        
    return df

@beartype
def normalize_cn_change(cn_change_table:pd.DataFrame, sample_counts:int) -> pd.DataFrame:
    """ Create columns for gains and losses divided by sample count
    """
    cn_change_table['norm_gain'] = cn_change_table['gain'] / sample_counts
    cn_change_table['norm_loss'] = cn_change_table['loss'] / sample_counts
    return cn_change_table

@beartype
def make_gene_pos_table(gene_list:set) -> dict:
    """ Parse refFlat [external] and return dict with gene -> (chrom, cds_start) tuple
    """
    refflat_path = "/home/chois7/ondemand/spectrumanalysis/analysis/notebooks/bulk-dna/data/refFlat.txt.gz"
    df = pd.read_table(refflat_path, names=['symbol', 'transcript', 'chromosome', 'strand',
                                            'cdna_start', 'cdna_end', 'cds_start', 'cds_end', 
                                            'exon_cnt', 'exon_starts', 'exon_ends',])
    gene_pos_table = {}
    for gene in gene_list:
        sub_df = df.loc[df['symbol']==gene, :]
        cds_length = sub_df.loc[:,'cds_end'] - sub_df.loc[:,'cds_start']
        row = sub_df[cds_length == cds_length.max()].iloc[0,:]
        gene_pos_table[gene] = (str(row.chromosome).replace('chr',''), int(row.cds_start)) # PIK3CA -> 178916613
        logging.debug(f'{row.symbol}, {row.cds_start}')
    return gene_pos_table

def get_norm_values_and_color(df:pd.DataFrame, chrom:str, pos:int, bin_size=500000) -> tuple:
    """ Get (y-coord, color) for normalized aggregate CN counts
    """
    weight = 1.5 # ad hcc
    offset = 0.1 # ad hcc
    max_y_value = 0.85 # ad hcc

    pos = int(pos / bin_size) * bin_size
    row = df.loc[
        (df.chromosome == chrom) &
        (df.start == pos)
    ]
    norm_gain = float(row.norm_gain)
    norm_loss = float(row.norm_loss)
    if (abs(norm_gain) > abs(norm_loss)):
        color = 'red'
        y_value = min(max_y_value, norm_gain * weight + offset)
    else:
        color = 'blue'
        y_value = -1 * min(max_y_value, norm_loss * weight + offset)

    return (y_value, color)


@beartype
def plot_cnv_segments(ax:matplotlib.axes.Axes, cnv:pd.DataFrame, 
        column:str, segment_color:tuple, fill=False) -> None:
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

    cnv = cnv.sort_values('start')

    segments = wgs_analysis.plots.cnv.create_segments(cnv, column)
    ax.add_collection(matplotlib.collections.LineCollection(segments, colors=segment_color, lw=1))

    connectors = wgs_analysis.plots.cnv.create_connectors(cnv, column)
    ax.add_collection(matplotlib.collections.LineCollection(connectors, colors=segment_color, lw=1))

    if fill:
        quads = wgs_analysis.plots.cnv.create_quads(cnv, column)
        ax.add_collection(matplotlib.collections.PolyCollection(quads, facecolors=segment_color, edgecolors=segment_color, lw=0))

@beartype
def plot_cnv_genome(ax:matplotlib.axes.Axes, cnv:pd.DataFrame, chroms:list,
        col:str, color:tuple, fontprop:matplotlib.font_manager.FontProperties, 
        scatter=False, squashy=False) -> None:
    """
    Plot major/minor copy number across the genome

    Args:
        ax (matplotlib.axes.Axes): plot axes
        col (str): name of column to use

    """

    cnv = cnv.copy()

    if 'length' not in cnv:
        cnv['length'] = cnv['end'] - cnv['start'] + 1

    cnv = cnv[['chromosome', 'start', 'end', 'length', col]]

    cnv = cnv[cnv['chromosome'].isin(wgs_analysis.refgenome.info.chromosomes)]

    cnv.set_index('chromosome', inplace=True)
    cnv['chromosome_start'] = wgs_analysis.refgenome.info.chromosome_start
    cnv.reset_index(inplace=True)

    cnv['start'] = cnv['start'] + cnv['chromosome_start']
    cnv['end'] = cnv['end'] + cnv['chromosome_start']

    plot_cnv_segments(ax, cnv, column=col, segment_color=color, fill=True)

    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim((-0.5, wgs_analysis.refgenome.info.chromosome_end[chroms].max()))
    ax.set_xlabel('Chromosome', fontproperties=fontprop, fontsize=20) #
    ax.set_ylabel('CN event frequency', fontproperties=fontprop, fontsize=20) #
    ax.set_xticks([0] + list(wgs_analysis.refgenome.info.chromosome_end[chroms].values))
    plt.yticks(fontproperties=fontprop)
    ax.set_xticklabels([])
    
    ax.tick_params(labelsize=15)

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(wgs_analysis.refgenome.info.chromosome_mid[chroms]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(chroms))

    ax.xaxis.grid(True, which='major', linestyle=':')
    ax.yaxis.grid(True, which='major', linestyle=':')

@beartype
def plot_cn_change(cn_change:pd.DataFrame, chroms:list, fontprop:matplotlib.font_manager.FontProperties) -> tuple:
    fig = plt.figure(figsize=(17, 6))
    ax = fig.add_subplot(111)

    color_gain = plt.get_cmap('RdBu')(0.1)
    color_loss = plt.get_cmap('RdBu')(0.9)

    plot_cnv_genome(ax, cn_change, 
            chroms, 'norm_gain', color_gain, fontprop=fontprop)
    plot_cnv_genome(ax, cn_change.assign(loss=lambda df: -df['norm_loss']), 
            chroms, 'loss', color_loss, fontprop=fontprop)

    y_low_lim = -1 #ad hoc #-cn_change['norm_loss'].max() 
    y_high_lim = 1 #ad hoc #cn_change['norm_gain'].max() 
    plt.ylim((y_low_lim, y_high_lim))
    _ = ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) # format y-axis as -100~100%
    
    ax.tick_params(axis='x', labelsize=4)
    return fig, ax

@beartype
def calc_gene_x_coord(gene_chrom:str, gene_pos:int, 
        chroms:list, chrom_size_table:dict, bin_size=500000) -> int:
    """ Calculate x-coordinate of gene text on pan-chromosome plot
    """
    chrom_size_sum = 0
    for chrom in chroms:
        if chrom == gene_chrom:
            chrom_size_sum += gene_pos
            break
        else:
            chrom_size = chrom_size_table[chrom]
            chrom_size_sum += chrom_size
        chrom_size_sum = int(chrom_size_sum / bin_size) * bin_size
    return chrom_size_sum

@beartype
def get_chrom_arm_coords(centromere_table:pd.DataFrame) -> pd.DataFrame:
    """ Get cumulative start-end positions for p- and q-arms
    """
    arms = pd.DataFrame(columns=['chromosome', 'start', 'end', 'length', 'tag'])
    for rix, row in centromere_table.iterrows():
        chrom = row['chromosome']
        centromere_mid = (row['start'] + row['end']) // 2
        cum_start = wgs_analysis.refgenome.info.chromosome_start[chrom]
        cum_mid = cum_start + centromere_mid
        cum_end = wgs_analysis.refgenome.info.chromosome_end[chrom]
        p_arm = pd.DataFrame({
            'chromosome': [chrom],
            'start': [cum_start],
            'end': [cum_mid],
            'length': [cum_mid - cum_start],
            'tag': ['p'],
        })
        q_arm = pd.DataFrame({
            'chromosome': [chrom],
            'start': [cum_mid],
            'end': [cum_end],
            'length': [cum_end - cum_mid],
            'tag': ['q'],
        })
        arms = pd.concat([arms, p_arm, q_arm])
    arms.sort_values(by=['start'], inplace=True)
    return arms

@beartype
def make_p_arm_rectangle(chrom:str, arms:pd.DataFrame) -> plt.Rectangle:
    """ Create pyplot Rectangle for the p-arm
    """
    px0 = arms.loc[(arms['chromosome']==chrom) & (arms['tag']=='p'), 'start'].squeeze()
    width = arms.loc[(arms['chromosome']==chrom) & (arms['tag']=='p'), 'length'].squeeze()
    py0 = -1
    height = 2
    color = 'grey'
    p_arm = plt.Rectangle((px0, py0), width, height, color=color, alpha=0.07)
    return p_arm

@beartype
def get_gene_ranges(gene_list:list) -> dict:
    """ Return dict[gene] -> (chrom, cDNA start, cDNA end) from refFlat [external]
    """
    refflat_path = "/home/chois7/ondemand/spectrumanalysis/analysis/notebooks/bulk-dna/data/refFlat.txt.gz"
    rf = pd.read_table(refflat_path, names=['symbol', 'transcript', 'chromosome', 'strand',
                                            'cdna_start', 'cdna_end', 'cds_start', 'cds_end',
                                            'exon_cnt', 'exon_starts', 'exon_ends',])
    gene_ranges = {}
    for gene in gene_list:
        sub_df = rf.loc[rf['symbol']==gene, :]
        cdna_length = sub_df.loc[:,'cdna_end'] - sub_df.loc[:,'cdna_start']
        isoform = sub_df[cdna_length == cdna_length.max()].iloc[0,:]
        gene_ranges[gene] = (str(isoform.chromosome).replace('chr',''), 
                             int(isoform.cdna_start), 
                             int(isoform.cdna_end)) # PIK3CA -> 178916613
    return gene_ranges

@beartype
def get_gene_set(gene_list_file:str) -> set:
    """ Parse gene list from gene list file
    """
    assert os.path.exists(gene_list_file), gene_list_file
    gene_list = set([g.strip() for g in open(gene_list_file, 'r').readlines()])
    return gene_list

@beartype
def get_arial_fonts() -> tuple:
    """ Return tuple of Arial font and Arial Italic font
    """
    path = "/juno/work/shah/users/chois7/envs/p37/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf"
    fontprop = font_manager.FontProperties(fname=path)
    plt.rcParams['font.family'] = fontprop.get_name()
    italic_path = "/juno/work/shah/users/chois7/envs/p37/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/Arial-Italic.ttf"
    italic_fontprop = font_manager.FontProperties(fname=italic_path)
    return fontprop, italic_fontprop

@beartype
def get_cn_change(cn_data:pd.DataFrame, chroms:list) -> pd.DataFrame:
    """ Finalize CN change ready for plotting by creating, filtering, normalizing, and labeling it
    """
    sample_counts = cn_data['isabl_sample_id'].unique().shape[0]
    data, ploidy = get_per_sample_data_and_ploidy(cn_data)
    mean_above_background = get_mean_above_background(data)
    cn_change_table = create_cn_change_table(data, ploidy, mean_above_background)

    centromere_table = get_chromosome_gap_band_data()
    chrom_size_table = get_chrom_size_table(chroms)

    cn_change_table = normalize_cn_change(cn_change_table, sample_counts)

    filled_table = get_filled_table(cn_change_table, chrom_size_table, chroms=chroms, bin_size=500000)
    plot_data = label_centromere_regions(filled_table, centromere_table, chroms=chroms, bin_size=500000)
    plot_data.loc[plot_data['centromere'], ['norm_gain', 'norm_loss']] = 0.0 # remove CN from centromeric regions
    # plot_data[cohort].to_csv(args.out_table, sep='\t', index=False)
    return plot_data

@beartype
def get_metacohort_spectrum_signatures(cn_data:pd.DataFrame) -> pd.DataFrame:
    """ Return dataframe of ['sample', 'signature'] for SPECTRUM [external] and Metacohort [external]
    """
    spectrum_signature_path = '/juno/work/shah/users/chois7/spectrum/dlp/cn/mutational_signatures.tsv'
    metacohort_signature_path = '/juno/work/shah/users/chois7/metacohort/mmctm/resources/Tyler_2022_Nature_labels.isabl.tsv'
    columns = ['sample', 'signature']
    spectrum_ix = cn_data['isabl_sample_id'].str.startswith('SPECTRUM')
    cn_data.loc[spectrum_ix, 'sample'] = cn_data.loc[spectrum_ix, 'isabl_sample_id'].str.slice(0, 15)
    cn_data.loc[~spectrum_ix, 'sample'] = cn_data.loc[~spectrum_ix, 'isabl_sample_id']
    spectrum_signature = pd.read_table(spectrum_signature_path)[['patient_id', 'consensus_signature']]
    spectrum_signature.columns = columns
    metacohort_signature = pd.read_table(metacohort_signature_path)[['isabl_sample_id', 'stratum']]
    metacohort_signature.columns = columns
    signatures = pd.concat([spectrum_signature, metacohort_signature])
    return signatures

@beartype
def create_cohort_cn_data(cohort:str) -> pd.DataFrame:
    """ Load get_cohort_cn cohort, filter, and return cohort CN dataframe
    """
    cn_data = shahlabdata.wgs.get_cohort_cn(cohort, most_recent=True)
    include_wgs_samples = get_include_wgs_samples(cohort)
    if len(include_wgs_samples) > 0:
        cn_data = cn_data[cn_data['isabl_aliquot_id'].isin(include_wgs_samples)]
    isabl_aliquots = cn_data['isabl_aliquot_id'].unique()
    qc_cn_data_and_samples(cn_data)

    cn_data = select_histotype_and_project(cn_data, cohort)
    cn_data = pd.concat([cn_data.assign(cohort=cohort)])
    return cn_data

@beartype
def make_merged_cn_data(cohorts=['Metacohort', 'SPECTRUM']) -> tuple:
    """ Return merged melted CN per bin dataframe for given cohorts, with signature labels, and sample counts dict
    """
    chroms = wgs_analysis.refgenome.info.chromosomes[:-1]
    os.environ['ISABL_API_URL'] = 'https://isabl.shahlab.mskcc.org/api/v1'
    os.environ['ISABL_CLIENT_ID'] = '3'

    saved = {
        'Metacohort': '/juno/work/shah/users/chois7/metacohort/wgs/cn/Metacohort.cn_data.tsv',
        'SPECTRUM': '/juno/work/shah/users/chois7/spectrum/wgs/cn/SPECTRUM.cn_data.tsv',
        'merged': '/juno/work/shah/users/chois7/tickets/cohort-cn-qc/results/merged.cn_data.tsv'
    }

    if cohorts.count('SPECTRUM DLP') > 0:
        logging.info('data retrieval for SPECTRUM DLP not ready')
        return pd.DataFrame()

    if os.path.exists(saved['merged']):
        cn_data = pd.read_table(saved['merged'])
    else:
        cn_merged = pd.DataFrame()
        for cohort in cohorts:
            if os.path.exists(saved[cohort]):
                cn_data = pd.read_table(saved[cohort])
                logging.debug(f'{cohort} data retrieved: cohort.shape: {cn_data.shape}')
            else:
                cn_data = create_cohort_cn_data(cohort)
                cn_data.to_csv(saved[cohort], sep='\t', index=False)

            cn_merged = pd.concat([cn_merged, cn_data])
        cn_data = cn_merged.copy()

    signatures = get_metacohort_spectrum_signatures(cn_data)
    cn_data = cn_data.merge(signatures)
    signature_counts = {}
    for signature in signatures['signature'].unique():
        signature_count = cn_data[cn_data['signature']==signature]['isabl_sample_id'].unique().shape[0]
        signature_counts[signature] = signature_count
    signature_counts['merged'] = cn_data['isabl_sample_id'].unique().shape[0]

    cn_changes = {}
    signatures = cn_data['signature'].unique()
    for signature in signatures:
        logging.info(f'signature: {signature}')
        signature_cn_data = cn_data[cn_data['signature']==signature]
        cn_changes[signature] = get_cn_change(signature_cn_data, chroms)
    cn_changes['merged'] = get_cn_change(cn_data, chroms)
    
    return cn_changes, signature_counts


class CopyNumberChangeData:
    def __init__(self, cohorts=['Metacohort', 'SPECTRUM'], gene_list=None,
            bin_size=500000, factor=1000000):
        self.cohorts = cohorts
        self.fontprop, self.italic_fontprop = get_arial_fonts()
        self.chroms = wgs_analysis.refgenome.info.chromosomes[:-1]
        self.chrom_size_table = get_chrom_size_table(self.chroms)
        if type(gene_list) == str:
            #gene_list_file = '/juno/work/shah/users/chois7/tickets/cohort-cn-qc/resources/gene_list.txt'
            assert os.path.exists(gene_list), gene_list
            self.gene_list = get_gene_set(gene_list)
        elif isinstance(gene_list, Iterable):
            self.gene_list = set(gene_list)
        else:
            raise TypeError(f'Type of gene_list is neither str or Iterable: {gene_list}')

        self.gene_pos_table = make_gene_pos_table(self.gene_list)
        self.cn_changes, self.signature_counts = make_merged_cn_data(self.cohorts)
        self.sample_counts = self.signature_counts['merged']
        self.centromere_table = get_chromosome_gap_band_data()
        self.arms = get_chrom_arm_coords(self.centromere_table)
        self.bin_size = bin_size
        self.factor = factor
        #self.out_path = 

    def plot_pan_chrom_cn(self, group='merged', out_path=None):
        cohort_symbol = ' + '.join(self.cohorts)
        plot_title = f'{cohort_symbol} (n={self.sample_counts})'
        with matplotlib.rc_context({'font.family':'Arial', 'font.size': 15}):
            # Draw GISTIC summary plot
            plot_data = self.cn_changes[group]
            fig, ax = plot_cn_change(plot_data, self.chroms, fontprop=self.fontprop)
            ax.set_title(plot_title, fontproperties=self.fontprop, fontsize=22) #

            # Add p- and q-arm rectangles
            for chrom in self.chroms:
                p_arm = make_p_arm_rectangle(chrom, self.arms)
                ax.add_patch(p_arm)

            # Add label per gene
            for gene in self.gene_list:
                gene_chrom, gene_pos = self.gene_pos_table[gene]
                gene_x_pos = calc_gene_x_coord(gene_chrom, gene_pos, 
                        chroms=self.chroms, chrom_size_table=self.chrom_size_table, bin_size=self.bin_size)
                (gene_y_value, color) = get_norm_values_and_color(plot_data, gene_chrom, gene_pos)
                logging.debug(f'{gene} - x:{gene_x_pos}, y:{gene_y_value} - {color}')
                if gene in ('BRCA1', 'RB1', 'TP53',):
                    gene_y_value *= 1.1 # ad hoc
                ax.bar(gene_x_pos, gene_y_value, width=3000000, color="gray", zorder=3)
                ax.annotate('%s' % gene, (gene_x_pos, gene_y_value * 1.05), ha='center', va='center', # non-italic
                             fontproperties=self.italic_fontprop, fontsize=15)

            # Save plot to png
            if out_path:
                plt.savefig(out_path)

    def plot_per_chrom_cn(self, out_path):
        cohort_symbol = ' + '.join(self.cohorts)
        plot_title = f'{cohort_symbol} (n={self.sample_counts})'
        

logging.basicConfig(level = logging.INFO)
gene_list_path = '/juno/work/shah/users/chois7/tickets/cohort-cn-qc/resources/gene_list.txt'
for cohorts in (['SPECTRUM'], ['Metacohort'], ['SPECTRUM', 'Metacohort']):
    cn = CopyNumberChangeData(gene_list=gene_list_path, cohorts=cohorts)
    cohort_symbol = '_'.join(cn.cohorts)
    for signature in cn.signature_counts:
        logging.info(f'processing cohorts:{cohorts} signature:{signature}')
        if signature != 'merged' and cn.signature_counts[signature] > 3:
            cn.plot_pan_chrom_cn(group=signature, out_path=f'{cohort_symbol}.{signature}.pdf')
