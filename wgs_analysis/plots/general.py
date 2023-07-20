'''
# General plot functions

'''

import pysam
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from .. import refgenome

# import wgs_analysis.helpers as helpers

# def plot_patient_legend(legend_filename):
#     fig = plt.figure(figsize=(0.75, 2))
#     ax = fig.add_subplot(111)
#     ax.axis('off')

#     artists = [plt.Circle((0, 0), color=c) for c in helpers.patient_cmap.values()]
#     ax.legend(artists, helpers.patient_cmap.keys(), loc='upper right', title='Patient')

#     fig.savefig(legend_filename, bbox_inches='tight')


def convert_tupleproxy_to_pyranges(exon):
    """ Convert TabixFile 'exon' (TupleProxy) to pyranges-like fields
    - exon: TupleProxy (result from iterating TabixFile
    
    Returns: pyranges-like fields [list]
    """
    assert len(exon) == 9, f'exon (length:{len(exon)}): {tuple(exon)}'
    exon_field = exon[:8]
    info = exon[8].rstrip(';')
    annots = info.split('; ')
    annot_map = {v.split(' ')[0]: v.split(' ')[1][1:-1] for v in annots} # [1:-1] hack to quick-remove ""

    annot_cols = ['gene_id', 'gene_version', 'gene_name', 'gene_source',
        'gene_biotype', 'transcript_id', 'transcript_version',
        'transcript_name', 'transcript_source', 'transcript_biotype', 'tag',
        'transcript_support_level', 'exon_number', 'exon_id', 'exon_version',
        'protein_id', 'protein_version', 'ccds_id']

    annot_field = [annot_map[k] if k in annot_map else np.nan for k in annot_cols]
    field = exon_field + annot_field
    return field

def parse_gtf_region(gtf, region):
    """ Convert pysam TabixFile fetch results to quasi-pyranges DataFrame
    - gtf: pysam.TabixFile
    - region: tuple of (chromosome[str], start[int], end[int])
    
    Returns: pd.DataFrame
    """
    pyranges_cols = [
        'Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand',
        'Frame', 'gene_id', 'gene_version', 'gene_name', 'gene_source',
        'gene_biotype', 'transcript_id', 'transcript_version',
        'transcript_name', 'transcript_source', 'transcript_biotype', 'tag',
        'transcript_support_level', 'exon_number', 'exon_id', 'exon_version',
        'protein_id', 'protein_version', 'ccds_id'
    ]
    int_cols = ['Start', 'End',]
    df = pd.DataFrame(columns=pyranges_cols)
    for exon in gtf.fetch(*region, parser=pysam.asTuple()):
        field = convert_tupleproxy_to_pyranges(exon)
        df.loc[df.shape[0]] = field
    df[int_cols] = df[int_cols].astype(int)
    return df

def get_gene_repr_exons(gtf, gene, lenient=False):
    """ Get representative exons from given gene name
    - gtf: pyranges-like DataFrame from parsing GTF
    - gene: gene symbol

    Returns: DataFrame of exons of the representative isoform of the give gene
    """
    transcript = get_repr_transcript_id(gtf, gene, lenient=lenient)
    exons = get_transcript_exons(gtf, transcript)
    return exons

def get_transcript_exons(gtf, transcript_id):
    exons = gtf[
        (gtf['transcript_id']==transcript_id) &
        (gtf['Feature'].isin(['exon', 'CDS']))
    ]
    return exons

def get_repr_transcript_id(gtf, gene_name, lenient=False):
    """ Select the representative transcript isoform of a gene symbol
    - gtf: pyranges-like DataFrame from parsing GTF
    - gene: gene symbol

    Returns: transcript_id[str]
    """
    if not lenient:
        transcripts = (
            gtf
            .query(f"gene_name == '{gene_name}'")
            .query("Feature == 'transcript'")
            .query("transcript_biotype == 'protein_coding'")
            .query("transcript_support_level == '1'")
        ).copy()
    else:
        transcripts = (
            gtf
            .query(f"gene_name == '{gene_name}'")
            .query("Feature == 'transcript'")
        ).copy()
    transcript_id = None
    if transcripts.shape[0] > 0:
        transcripts['length'] = transcripts['End'] - transcripts['Start']
        transcript = transcripts.sort_values(by=['length'], ascending=False).iloc[0]
        transcript_id = transcript['transcript_id']
    return transcript_id

def _get_gene_intervals(gene_exons):
    # gather gene start-end intervals
    intervals = []
    for gene, exons in gene_exons.items():
        start_min = exons['Start'].min()
        end_max = exons['End'].max()
        if not np.isnan(start_min) and not np.isnan(end_max):
            interval = (start_min, end_max, gene)
            intervals.append(interval)
    intervals.sort(key=lambda x: (x[0], x[1]))
    return intervals

def _get_gene_offsets(intervals):
    # set gene offset according to overlapping intervals
    gene2offset = defaultdict(int)
    n_intervals = len(intervals)
    margin = 5e4
    for i in range(n_intervals):
        for j in range(i+1, n_intervals):
            start_i, end_i, gene_i = intervals[i]
            start_j, end_j, gene_j = intervals[j]
            if is_overlap((start_i-margin, end_i+margin), (start_j-margin, end_j+margin)):
                if gene2offset[gene_i] == gene2offset[gene_j]:
                    gene2offset[gene_j] += 1
    return gene2offset

def add_gene_annotations(ax, chromosome, start, end, genes=None, return_offset=False):
    """ Plot UCSC-like gene annotation to Axes
    - ax: pyplot.Axes
    - gene_exons[dict]: {gene_symbol: pyranges-like GTF DataFrame}
    - return_offset: return gene -> offset map [dict]

    Returns: gene2offset[dict] (optional)
    """
    ax.set_xlim((start, end))
    ax.spines[['top', 'bottom', 'right', 'left']].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # setup which genes to plot
    region = (chromosome, start, end)
    gene_df = parse_gtf_region(refgenome.info.gtf, region)
    if genes == None: genes = gene_df['gene_name'].unique() # if no genes put in, use all genes in region
    gene_exons = {gene: get_gene_repr_exons(gene_df, gene, lenient=False) 
                  for gene in genes}
    intervals = _get_gene_intervals(gene_exons)
    gene2offset = _get_gene_offsets(intervals)
    
    # apply offset and plot exon rectangles
    for gene, exons in gene_exons.items():
        if exons.shape[0] == 0: continue # filter out if none
        strands = exons['Strand'].unique()
        assert len(strands) == 1, strands
        strand = strands[0]
        transcript_s, transcript_e = exons['Start'].min(), exons['End'].max()
        offset = gene2offset[gene]
        for i, (_, row) in enumerate(exons.iterrows()):
            add_annot = (i==0)
            _add_transcript_rectangles(ax, row, transcript_s, transcript_e, offset, strand, add_annot=add_annot) # add rectangles and a line

    if return_offset:
        return gene2offset

def _add_quiver(ax, x_quiver_starts, y_quiver_starts, x_quiver_directs, y_quiver_directs, box_color='#090A76'):
    ax.quiver(
        x_quiver_starts, y_quiver_starts, x_quiver_directs, y_quiver_directs,
        color=box_color,
        width=0.005, 
        headwidth=30, 
        headaxislength=5,
        headlength=17, 
        pivot='tip',
        scale=100,
    )

def _add_transcript_rectangles(ax, row, transcript_s, transcript_e, y_offset, strand, add_annot=False):
    box_color = '#090A76'
    strand_color = 'blue'
    x_direct, y_direct = -1, 0
    if strand == '+': 
        x_direct = 1
        strand_color = 'red'
    x_min, x_max = ax.get_xlim()
    y_offset_factor = 3
    y_offset *= y_offset_factor
    exon_s, exon_e = row['Start'], row['End']
    feature = row['Feature']
    rect_y, rect_height = 1+y_offset, 1
    transcript_line_y = rect_y + 0.5 * rect_height
    text_y = rect_y - rect_height + 3
    if feature == 'CDS':
        rect_y = 0.5 + y_offset
        rect_height = 2
    exon_len = exon_e - exon_s
    rect_x = exon_s #max(exon_s, start)
    
    # for each exon / CDS
    ax.add_patch(matplotlib.patches.Rectangle((rect_x, rect_y), exon_len, rect_height, 
                                              linewidth=1, edgecolor=strand_color, color=box_color, zorder=1))
    if add_annot: 
        quiver_interval = int((x_max - x_min) / 50)
        x_quiver_starts = np.arange(transcript_s, transcript_e, quiver_interval)[1:-1]
        n_quivers = len(x_quiver_starts)
        y_quiver_starts = [transcript_line_y] * n_quivers
        x_quiver_directs = [x_direct] * n_quivers
        y_quiver_directs = [y_direct] * n_quivers
        _add_quiver(ax, x_quiver_starts, y_quiver_starts, x_quiver_directs, y_quiver_directs)
        ax.plot([transcript_s+50, transcript_e-100], 
                [transcript_line_y, transcript_line_y], 
                linewidth=1, color=box_color, zorder=0)
        text_x = (transcript_s + transcript_e) / 2
        if text_x < x_min:
            text_x = x_min
        if text_x > x_max:
            text_x = x_max
        ax.text(x=text_x, y=text_y, s=row['gene_name'], color=box_color, horizontalalignment='center', fontdict={'size':11})

# cytoband tags (genomic position)
_gtags = [s.replace('\n', '') for s in '''
gpos1,gpos2,gpos3,gpos4,gpos5,gpos6,gpos7,gpos8,
gpos9,gpos10,gpos11,gpos12,gpos13,gpos14,gpos15,gpos16,
gpos17,gpos18,gpos19,gpos20,gpos21,gpos22,gpos23,gpos24,
gpos25,gpos26,gpos27,gpos28,gpos29,gpos30,gpos31,gpos32,
gpos33,gpos34,gpos35,gpos36,gpos37,gpos38,gpos39,gpos40,
gpos41,gpos42,gpos43,gpos44,gpos45,gpos46,gpos47,gpos48,
gpos49,gpos50,gpos51,gpos52,gpos53,gpos54,gpos55,gpos56,
gpos57,gpos58,gpos59,gpos60,gpos61,gpos62,gpos63,gpos64,
gpos65,gpos66,gpos67,gpos68,gpos69,gpos70,gpos71,gpos72,
gpos73,gpos74,gpos75,gpos76,gpos77,gpos78,gpos79,gpos80,
gpos81,gpos82,gpos83,gpos84,gpos85,gpos86,gpos87,gpos88,
gpos89,gpos90,gpos91,gpos92,gpos93,gpos94,gpos95,gpos96,
gpos97,gpos98,gpos99,gpos100,gneg,acen,gvar,stalk
'''.strip().split(',')]

# cytoband colors
_gcolors = [
    "#FDFDFD", "#FBFBFB", "#F8F8F8", "#F6F6F6", "#F3F3F3", "#F1F1F1", "#EEEEEE", "#ECECEC", 
    "#E9E9E9", "#E6E6E6", "#E4E4E4", "#E1E1E1", "#DFDFDF", "#DCDCDC", "#DADADA", "#D7D7D7", 
    "#D4D4D4", "#D2D2D2", "#CFCFCF", "#CDCDCD", "#CACACA", "#C8C8C8", "#C5C5C5", "#C3C3C3", 
    "#C0C0C0", "#BDBDBD", "#BBBBBB", "#B8B8B8", "#B6B6B6", "#B3B3B3", "#B1B1B1", "#AEAEAE", 
    "#ACACAC", "#A9A9A9", "#A6A6A6", "#A4A4A4", "#A1A1A1", "#9F9F9F", "#9C9C9C", "#9A9A9A", 
    "#979797", "#949494", "#929292", "#8F8F8F", "#8D8D8D", "#8A8A8A", "#888888", "#858585", 
    "#838383", "#808080", "#7D7D7D", "#7B7B7B", "#787878", "#767676", "#737373", "#717171", 
    "#6E6E6E", "#6C6C6C", "#696969", "#666666", "#646464", "#616161", "#5F5F5F", "#5C5C5C", 
    "#5A5A5A", "#575757", "#545454", "#525252", "#4F4F4F", "#4D4D4D", "#4A4A4A", "#484848", 
    "#454545", "#434343", "#404040", "#3D3D3D", "#3B3B3B", "#383838", "#363636", "#333333", 
    "#313131", "#2E2E2E", "#2C2C2C", "#292929", "#262626", "#242424", "#212121", "#1F1F1F", 
    "#1C1C1C", "#1A1A1A", "#171717", "#141414", "#121212", "#0F0F0F", "#0D0D0D", "#0A0A0A", 
    "#080808", "#050505", "#030303", "#000000", "#FFFFFF", "#660033", "#9966EE", "#6600CC", 
]


def get_cytobands_dataframe(genome_version, _gtags=_gtags, _gcolors=_gcolors):
    """ Return cytobands DataFrame with the following shape:

        chromosome │ start     │ end       │ name   │ gtag    │ color   ║
        1          │ 0         │ 2300000   │ p36.33 │ gneg    │ #FFFFFF ║
        1          │ 2300000   │ 5300000   │ p36.32 │ gpos25  │ #C0C0C0 ║
        1          │ 5300000   │ 7100000   │ p36.31 │ gneg    │ #FFFFFF ║
        1          │ 7100000   │ 9100000   │ p36.23 │ gpos25  │ #C0C0C0 ║
        1          │ 9100000   │ 12500000  │ p36.22 │ gneg    │ #FFFFFF ║
        ...	...	...	...	...	...	...

        - genome_version: genome version, supported = {"hg19", "hg38", "grch38"}
    """
    import gzip
    import requests
    
    cytobands_paths = {
        'hg19': 'https://s3.amazonaws.com/igv.broadinstitute.org/genomes/seq/hg19/cytoBand.txt',
        'hg38': 'https://s3.amazonaws.com/igv.org.genomes/hg38/annotations/cytoBandIdeo.txt.gz',
        'grch38': 'https://s3.amazonaws.com/igv.org.genomes/hg38/annotations/cytoBandIdeo.txt.gz'
    }

    cytobands = None
    url = cytobands_paths[genome_version]
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        cytobands_cols = ['chromosome', 'start', 'end', 'name', 'gtag']
        cytobands = pd.DataFrame(columns=cytobands_cols)
        if url.endswith('.gz'):
            gzip_file = gzip.GzipFile(fileobj=response.raw)
            text = gzip_file.read().decode('utf-8')
        else:
            text = response.text
        fields = [line.split('\t') for line in text.split('\n')]
        for field in fields:
            if len(field) == len(cytobands_cols):
                cytobands.loc[cytobands.shape[0]] = field
        cytobands['chromosome'] = cytobands['chromosome'].str.replace('chr', '')
        _gtag2color = dict(zip(_gtags, _gcolors))
        cytobands['color'] = cytobands['gtag'].map(_gtag2color)
    else:
        print(f'ERROR: Failed to download the file in {url}')
        
    return cytobands


def add_cytobands_to_ax(ax, chromosome, start, end, genome_version, show_xaxis_tick_top,
        _gtags=_gtags, _gcolors=_gcolors):
    """ Plot cytobands to given Axes
    - ax: pyplot Axes
    - cytobands: cytobands DataFrame
    - chromosome: chromosome (either with chr prefix or not)
    - start: highlighting coordinate start
    - end: highlighting coordinate end
    - genome_version: reference genome version {"hg19", "hg38", "grch38"}
    - show_xaxis_tick_top: show xaxis ticks at top
    """
    cytobands = get_cytobands_dataframe(genome_version, _gtags=_gtags, _gcolors=_gcolors)
    if chromosome.startswith('chr'):
        chromosome = chromosome.replace('chr', '')

    ax.set_yticklabels([]); ax.set_yticks([]); ax.set_ylabel('')
    ax.set_xlabel('chr'+chromosome, fontsize=13)
    if show_xaxis_tick_top:
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
    
    linewidth = 3
    height = 1

    refgenome.set_genome_version(genome_version)
    chrom_end = refgenome.info.chromosome_lengths.loc[chromosome]
    ax.set_xlim((0, chrom_end))
    chrom_cytobands = cytobands[cytobands['chromosome'] == (chromosome.replace('chr', ''))]
    for _, row in chrom_cytobands.iterrows():
        cyto_start, cyto_end = int(row['start']), int(row['end'])
        cyto_color = row['color']
        length = cyto_end - cyto_start
        box = matplotlib.patches.FancyBboxPatch(xy=(cyto_start, 0), width=length, height=height, color=cyto_color,
                                                boxstyle="round,pad=0,rounding_size=0")
        ax.add_patch(box)
    
    x_span = end - start
    y_span = height
    x0 = start 
    width_adj = x_span
    y0 = 0 + y_span / 50
    height_adj = height - y_span / 40
    
    highlight = matplotlib.patches.Rectangle(xy=(x0, y0), width=width_adj, height=height_adj, zorder=2,
                                             edgecolor='red', facecolor='none', linewidth=linewidth, linestyle=(0, (1,1)))
    ax.add_patch(highlight)

# UCSC-like gene plotting
def is_overlap(interval1, interval2):
    """ Return overlap status of two intervals: test the following:
    interval1:   [   )
    interval2: [       )
    interval2:  [ )
    interval2:    [ )
    interval2:      [ )
     - otherwise no overlap
    """
    start1, end1 = interval1
    start2, end2 = interval2
    start1, end1, start2, end2 = float(start1), float(end1), float(start2), float(end2)
    overlap = (start2 < end1) and (start1 < end2)
    return overlap

