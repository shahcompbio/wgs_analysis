'''
# Single cell composite plots

'''

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle as Rectangle
from mpl_toolkits.axes_grid import Divider, LocatableAxes, Size
import seaborn

import wgs_analysis.helpers as helpers
import wgs_analysis.plots as plots
import wgs_analysis.plots.colors
import wgs_analysis.plots.utils
import wgs_analysis.plots.trees


def plot_tree_genotype_samples(tree, genotype_table, sample_table, prevalence_type='count', event_colors=None, small_values=None):
    """ Plot a genotype tree with sample specific genotype prevalences/counts

    Args:
        tree (dollo.trees.TreeNode): tree to plot
        genotype_table (pandas.DataFrame): table of genotypes per node
        sample_table (pandas.DataFrame): table of sample prevalences/counts

    KwArgs:
        prevalence_type (str): type of barplot, 'count' or 'proportion'
        event_colors (dict): event colors keyed by event id
        small_values (float): label positive values less than or equal this value

    The genotype table should have node labels as row indices and event ids (ids of 
    snv/breakpoint clusters) as column labels.  Each 0/1 entry represents the
    presence or absence of the event for that node.  The sample table should have
    sample ids as row indices and leaf names as column labels.

    """

    leaf_names = [str(leaf.name) for leaf in tree.leaves]
    events = [str(event) for event in genotype_table.columns]

    num_samples = len(sample_table.index)
    num_events = len(genotype_table.columns)

    if event_colors is None:
        if num_events > 8:
            palette = seaborn.color_palette('husl', num_events)
        else:
            palette = seaborn.color_palette('Set2', num_events)
        event_colors = dict(zip(genotype_table.columns.values, palette))

    fig = plt.figure(figsize=(20, 3))
    plots.utils.setup_plot()

    # Plot grid
    gs = gridspec.GridSpec(1, num_samples + 2, height_ratios=[1], width_ratios=[2,2] + [1] * num_samples)

    # Plot tree
    tree_ax = plt.subplot(gs[0])

    plots.trees.plot_tree(tree_ax, tree, extend_leaf_branch=True, landscape=True, flip=False)
    plots.trees.plot_branch_events(tree_ax, tree, genotype_table, event_colors=event_colors)
    plots.utils.shift_lims(tree_ax, 12, 12)

    tree_ax.set_yticks(())
    tree_ax.yaxis.set_tick_params(right='off')
    tree_ax.spines['right'].set_visible(False)
    tree_ax.set_title('Clone Phylogeny')

    # Genotype plots
    genotype_ax = plots.utils.setup_axes(plt.subplot(gs[1]))

    width = 1.0
    height = 0.8

    for leaf in tree.leaves:

        offset_x = 0
        offset_y = -height * 0.5

        for idx, (event_id, present) in enumerate(genotype_table.loc[leaf.label].iteritems()):

            widget_x = width * idx - offset_x
            widget_y = leaf.y + offset_y

            color = 'white'
            if present == 1:
                color = event_colors[event_id]

            genotype_ax.add_patch(Rectangle((widget_x, widget_y), width, height,
                facecolor=color, edgecolor='.75', linewidth=0.25, zorder=10))

        genotype_ax.add_patch(Rectangle((0, leaf.y + offset_y), num_events, height,
            facecolor=(0,0,0,0), edgecolor='k', linewidth=1., zorder=10))

    genotype_ax.set_xticks(np.arange(num_events) + 0.5)
    genotype_ax.set_xticklabels(events)
    genotype_ax.set_yticks(xrange(len(leaf_names)))
    genotype_ax.set_yticklabels(leaf_names, ha='left')
    genotype_ax.set_xlim((0, num_events))
    genotype_ax.xaxis.set_tick_params(which='both', top=False, bottom=False, labeltop=False, labelbottom=True, labelsize=6, pad=0)
    genotype_ax.yaxis.set_tick_params(which='both', right=False, left=False, labelright=False, labelleft=True, labelsize=10, pad=0)
    genotype_ax.spines['left'].set_visible(False)
    genotype_ax.spines['right'].set_visible(False)
    genotype_ax.spines['top'].set_visible(False)
    genotype_ax.spines['bottom'].set_visible(False)
    genotype_ax.grid(False)
    genotype_ax.set_title('Observed Genotypes')

    # Plot barplots
    max_cell_count = sample_table.max().max()

    for idx, (sample_id, row) in enumerate(sample_table.iterrows()):

        ax = plots.utils.setup_axes(plt.subplot(gs[idx+2]))

        row = row.reindex(index=leaf_names)

        ind = np.arange(len(row))

        ax.barh(ind, row.values, align='center', height=1.0, color='0.8')

        if small_values is not None:
            for i, v in zip(ind, row.values):
                if v <= small_values and v > 0:
                    ax.text(v+3, i, '{0}'.format(int(v)), fontsize=8, ha='center', va='center')

        ax.grid(False)
        ax.spines['bottom'].set_position(('outward', 0))
        ax.spines['left'].set_position(('outward', 0))

        if prevalence_type == 'count':
            ax.set_xlim((ax.get_xlim()[0], max_cell_count * 1.05))
        elif prevalence_type == 'proportion':
            ax.set_xlim((ax.get_xlim()[0], 1.0))
        ax.set_ylim((min(ind) - 0.5, max(ind) + 0.5))
        ax.yaxis.set_tick_params(left='off', labelleft='off')
        ax.set_title(sample_id)
        ax.set_ylabel('')

        if idx == len(sample_table.index) - 1:
            ax.set_xlabel(prevalence_type)
            if prevalence_type == 'count':
                ax.set_xticks(np.linspace(0, max_cell_count, 3))
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(8) 
            elif prevalence_type == 'proportion':
                ax.set_xticks((0., 0.5, 1.0))
        else:
            ax.set_xticks(np.linspace(0, max_cell_count, 3))
            ax.set_xticklabels([])
            ax.set_xlabel('')

    tree_ax.set_ylim(ax.get_ylim())
    genotype_ax.set_ylim(tree_ax.get_ylim())

    fig.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.0)

    return fig



def plot_tree_genotype(tree, genotype_table, event_colors=None, palette=None):
    """ Plot a genotype tree with sample specific genotype prevalences/counts

    Args:
        tree (dollo.trees.TreeNode): tree to plot
        genotype_table (pandas.DataFrame): table of genotypes per node

    KwArgs:
        event_colors (dict): event colors keyed by event id
        use_same_colors (dict): color palette to use. If None will automatically choose based on number of genotypes. 

    The genotype table should have node labels as row indices and event ids (ids of 
    snv/breakpoint clusters) as column labels.  Each 0/1 entry represents the
    presence or absence of the event for that node.

    """

    leaf_names = [str(leaf.name) for leaf in tree.leaves]
    events = [str(event) for event in genotype_table.columns]

    num_leaves = len(leaf_names)
    num_events = len(genotype_table.columns)

    if event_colors is None:
        if palette is not None:
            palette = seaborn.color_palette(palette, num_events)
        
        else:
            if num_events > 8:
                palette = seaborn.color_palette('husl', num_events)
            else:
                palette = seaborn.color_palette('Set2', num_events)

        event_colors = dict(zip(genotype_table.columns.values, palette))

    fig = plt.figure()
    plots.utils.setup_plot()

    scale = 0.4

    horiz = [
        scale * num_leaves,   # tree
        0.2,                  # spacer
        scale * num_events,   # genotype
    ]

    vert = [
        scale * num_leaves,   # genotype/tree
    ]

    horiz = [Size.Fixed(a) for a in horiz]
    vert = [Size.Fixed(a) for a in vert]
    rect = (0.1, 0.1, 0.8, 0.8)

    tree_ax = fig.add_axes(rect, label='tree')
    genotype_ax = fig.add_axes(rect, label='genotype')

    divider = Divider(fig, rect, horiz, vert, aspect=False)

    tree_ax.set_axes_locator(divider.new_locator(nx=0, ny=0))
    genotype_ax.set_axes_locator(divider.new_locator(nx=2, ny=0))

    plots.utils.setup_axes(tree_ax)
    plots.utils.setup_axes(genotype_ax)

    # Plot tree

    plots.trees.plot_tree(tree_ax, tree, extend_leaf_branch=True, landscape=True, flip=False)
    plots.trees.plot_branch_events(tree_ax, tree, genotype_table, event_colors=event_colors)
    plots.utils.shift_lims(tree_ax, 12, 12)

    tree_ax.set_yticks(())
    tree_ax.yaxis.set_tick_params(right='off')
    tree_ax.spines['right'].set_visible(False)
    tree_ax.set_title('Phylogeny', y=0.9)

    width = 1.0
    height = 0.8

    for leaf in tree.leaves:

        offset_x = 0
        offset_y = -height * 0.5

        for idx, (event_id, present) in enumerate(genotype_table.loc[leaf.label].iteritems()):

            widget_x = width * idx - offset_x
            widget_y = leaf.y + offset_y

            color = 'white'
            if present == 1:
                color = event_colors[event_id]

            genotype_ax.add_patch(Rectangle((widget_x, widget_y), width, height,
                facecolor=color, edgecolor='.75', linewidth=0.25, zorder=10))

        genotype_ax.add_patch(Rectangle((0, leaf.y + offset_y), num_events, height,
            facecolor=(0,0,0,0), edgecolor='k', linewidth=1., zorder=10))

    genotype_ax.set_xticks(np.arange(num_events) + 0.5)
    genotype_ax.set_xticklabels(events)
    genotype_ax.set_xlabel('PyClone cluster')
    genotype_ax.set_yticks(xrange(len(leaf_names)))
    genotype_ax.set_yticklabels(leaf_names, ha='left')
    genotype_ax.set_xlim((0, num_events))
    genotype_ax.xaxis.set_tick_params(which='both', top=False, bottom=False, labeltop=False, labelbottom=True, labelsize=10, pad=-10)
    genotype_ax.yaxis.set_tick_params(which='both', right=False, left=False, labelright=False, labelleft=True, labelsize=10, pad=0)
    genotype_ax.spines['left'].set_visible(False)
    genotype_ax.spines['right'].set_visible(False)
    genotype_ax.spines['top'].set_visible(False)
    genotype_ax.spines['bottom'].set_visible(False)
    genotype_ax.grid(False)
    genotype_ax.set_title('Genotypes', y=0.9)

    tree_ax.set_ylim((-0.5, float(len(leaf_names)) + 0.5))
    genotype_ax.set_ylim((-0.5, float(len(leaf_names)) + 0.5))

    return fig



