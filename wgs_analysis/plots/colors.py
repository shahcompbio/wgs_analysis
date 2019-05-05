'''
# Color sets

Standardized color sets for consistent plots.

'''
from collections import OrderedDict

import brewer2mpl
import collections
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import wgs_analysis.plots as plots
import wgs_analysis.refgenome as refgenome


max_color_idx = 12

def get_color_set(labels):

    cmap = plt.get_cmap('Dark2')

    color_set = dict()
    for idx, label in enumerate(labels):
        assert idx <= max_color_idx
        color_set[label] = cmap(float(idx) / float(max_color_idx))

    return color_set


def create_chromosome_color_map():

    colors = sb.color_palette("muted")
    color_set = [colors[i % 2] for i in range(len(refgenome.info.chromosomes))]
    chromosome_color = collections.OrderedDict(zip(refgenome.info.chromosomes, color_set))
    return chromosome_color


def get_single_cell_color_set():
    snv_palette = brewer2mpl.get_map('YlGnBu', 'Sequential', 3).mpl_colors

    colors = OrderedDict((('A', snv_palette[0]),
                          ('AB', snv_palette[1]),
                          ('B', snv_palette[2]),
                          ('Unknown', 'gray'),
                          ('Absent', '#f0f0f0'),
                          ('Present', '#636363')))
    
    return colors

def get_data_frame_color_set(df, map_series='Set1', map_type='Qualitative'):
    '''
    Return a dictionary mapping values in data frame to colors in a color brewer set.
    '''
    values = sorted(pd.unique(df.values.ravel()))
    
    palette = sb.color_palette(map_series, len(values))

    colors = OrderedDict(zip(values, palette))
    
    return colors

def get_tree_node_color_set(patient_id, is_site_tree=True):
    if is_site_tree:
        tree = helpers.get_snv_tree(patient_id)
    
    else:
        tree = helpers.get_single_cell_tree(patient_id)
    
    node_labels = sorted([node.label for node in tree.nodes])

    color_set = get_color_set(node_labels)
    
    return OrderedDict(zip(node_labels, [color_set[x] for x in node_labels]))

def get_loss_deleted_colors():
    c = get_loss_deleted_color_map()
    
    return [c['Unexplained'], c['Deleted']]

def get_loss_deleted_color_map():
    return {'Unexplained' :'#1b9e77', 'Deleted' : '#d95f02'}

def add_loss_deleted_legend(ax, frame=True):
    color_map = get_loss_deleted_color_map()
    
    palette = color_map.values()

    artists = [plt.Circle((0, 0), color=c) for c in palette]
    
    legend = ax.legend(artists, color_map.keys(), loc='upper right', frameon=frame)

    if frame:
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('white')

    
    
