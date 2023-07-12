'''
# General plot functions

'''

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# import wgs_analysis.helpers as helpers

# def plot_patient_legend(legend_filename):
#     fig = plt.figure(figsize=(0.75, 2))
#     ax = fig.add_subplot(111)
#     ax.axis('off')

#     artists = [plt.Circle((0, 0), color=c) for c in helpers.patient_cmap.values()]
#     ax.legend(artists, helpers.patient_cmap.keys(), loc='upper right', title='Patient')

#     fig.savefig(legend_filename, bbox_inches='tight')


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

def calc_transcript_offset(transcript_start_ends, offset=0):
    curr_start, curr_end = transcript_start_ends[-1]
    prev_start, prev_end = transcript_start_ends[-2]
    if is_overlap((curr_start, curr_end), (prev_start, prev_end)):
        offset += 1
    return offset

def add_quiver(ax, x_quiver_starts, y_quiver_starts, x_quiver_directs, y_quiver_directs, box_color='#090A76'):
    ax.quiver(
        x_quiver_starts, y_quiver_starts, x_quiver_directs, y_quiver_directs,
        color=box_color,
        width=0.001, 
        headwidth=10, 
        headaxislength=3,
        headlength=6, 
        pivot='tip',
        scale=200,
    )

def add_transcript_rectangles(ax, row, transcript_s, transcript_e, y_offset, strand, add_annot=False):
    """ Add one exon rectangle, and if add_annot add central line, quivers, and gene_name text
    """
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
    rect_x = max(exon_s, x_min)
    
    # for each exon / CDS
    ax.add_patch(matplotlib.patches.Rectangle((rect_x, rect_y), exon_len, rect_height, 
                                                linewidth=1, edgecolor=strand_color, color=box_color, zorder=1))
    if add_annot: # done only once 
        # add central line
        ax.plot([transcript_s+50, transcript_e-100], [transcript_line_y, transcript_line_y], 
                linewidth=1, color=box_color, zorder=0)

        # add gene orientation quivers
        quiver_interval = int((x_max - x_min) / 60)
        x_quiver_starts = np.arange(transcript_s, transcript_e, quiver_interval)[1:-1]
        n_quivers = len(x_quiver_starts)
        y_quiver_starts = [transcript_line_y] * n_quivers
        x_quiver_directs = [x_direct] * n_quivers
        y_quiver_directs = [y_direct] * n_quivers
        add_quiver(ax, x_quiver_starts, y_quiver_starts, x_quiver_directs, y_quiver_directs, box_color)

        # add gene name text
        text_x = (transcript_s + transcript_e) / 2
        if text_x < x_min: text_x = x_min
        if text_x > x_max: text_x = x_max
        ax.text(x=text_x, y=text_y, s=row['gene_name'], color=box_color, ha='center', fontdict={'size':11})

def plot_gene_annotations(ax, gene_exons, return_offset=True):
    """ Plot UCSC-like gene annotations
    - ax: pyplot Axes
    - gene_exons: DataFrame of pygtf 'exon' and 'CDS' for the gene
    - return_offset: if True return gene2offset dict

    gene_exons is created by:
    [1] loading a gtf with pygtf
    [2] filtering gtf with a gene_name
    [3] selecting the transcript with largest transcript start-end span
    [4] filtering gtf from [2] with the transcript ID

    Returns:
    - gene2offset: dict[gene_name] = 
    """
    gene2offset = {}
    transcript_start_ends = []
    transcript_offset = 0
    for gene, exons in gene_exons.items():
        strands = exons['Strand'].unique()
        assert len(strands) == 1, strands
        strand = strands[0]
        transcript_s, transcript_e = exons['Start'].min(), exons['End'].max()
        transcript_start_ends.append((transcript_s, transcript_e))
        if len(transcript_start_ends) >= 2:
            transcript_offset = calc_transcript_offset(transcript_start_ends, offset=transcript_offset)
        gene2offset[gene] = transcript_offset
        for i, (_, row) in enumerate(exons.iterrows()):
            add_annot = (i==0)
            add_transcript_rectangles(ax, row, transcript_s, transcript_e, transcript_offset, strand, add_annot=add_annot) # add rectangles and a line
            
    if return_offset:
        return gene2offset