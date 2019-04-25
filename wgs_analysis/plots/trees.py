'''
# Tree plotting

'''
from matplotlib.patches import Rectangle as Rectangle

import dollo
import itertools
import numpy as np
import seaborn

import wgs_analysis.plots.utils


def add_depths(tree, depth):
    """ Add plot depths (y-axis) for noninformative branch lengths

    Args:
        tree (dollo.trees.TreeNode): tree to add depths
        depth (float): starting depth of subtree rooted at `tree`

    """

    tree.depth = depth
    for child in tree.children:
        add_depths(child, depth+1)


def add_depths_brlen(tree, branch_length_attr, parent_depth=0.):
    """ Add plot depths (y-axis) based on branch lengths.

    Args:
        tree (dollo.trees.TreeNode): tree to add depths
        branch_length_attr (str): tree node attribute to use as branch length for node/parent branch
        parent_depth (float): starting depth of subtree rooted at `tree`

    """

    tree.depth = parent_depth + getattr(tree, branch_length_attr)
    for child in tree.children:
        add_depths_brlen(child, branch_length_attr, tree.depth)
        

def add_positions(tree, leaf_cnt):
    """ Add plot positions (x-axis) for each node.

    Args:
        tree (dollo.trees.TreeNode): tree to add positions

    Leaves are evenly spaced.  Internal nodes are placed at the midpoint
    between the left and right-most positions of the children.

    """

    if len(tree.children) > 0:
        for child in tree.children:
            add_positions(child, leaf_cnt)
        posns = [child.position for child in tree.children]
        tree.position = (max(posns) + min(posns)) / 2.
    else:
        tree.position = float(next(leaf_cnt))


def plot_tree(ax, tree, branch_length_attr=None, leaf_name_attr=None, extend_leaf_branch=False, landscape=True, flip=False, **kwargs):
    """ Plot a tree.

    Args:
        ax (matplotlib.axes.Axes): plot axes
        tree (dollo.trees.TreeNode): tree to plot

    Kwargs:
        branch_length_attr (str): node attribute to use as branch length
        leaf_name_attr (str): node attribute to use as leaf name
        extend_leaf_branch (bool): extend leafs to axis when not using branch lengths
        landscape (bool): plot tree in landscape mode (evolution axis as x axis)
        flip (bool): flip the evolution axis
        **kwargs: additional kwargs to plot

    """

    if extend_leaf_branch and branch_length_attr is not None:
        raise ValueError('cannot extend leaf branches when given branch lengths')

    if 'linestyle' not in kwargs:
        kwargs['linestyle'] = '-'

    if 'color' not in kwargs:
        kwargs['color'] = 'k'

    tree.add_parent()
    
    if branch_length_attr is not None:
        add_depths_brlen(tree, branch_length_attr)
    else:
        add_depths(tree, 1)

    max_depth = max([node.depth for node in tree.nodes])

    if extend_leaf_branch:
        for node in tree.nodes:
            if node.is_leaf:
                node.depth = max_depth

    add_positions(tree, itertools.count())

    # Assign x,y positions to nodes
    for node in tree.nodes:
        if landscape:
            node.x = node.depth
            node.y = node.position
        else:
            node.x = node.position
            node.y = node.depth

    # Assign branch line segments to nodes
    for node in tree.nodes:
        parent_x, parent_y = 0, 0
        if node.parent is not None:
            parent_x, parent_y = node.parent.x, node.parent.y
        if landscape:
            node.branch = ((node.x, parent_x), (node.y, node.y))
        else:
            node.branch = ((node.x, node.x), (node.y, parent_y))

    # Create a collection of lateral edges to link node branches
    links = list()
    for node in tree.nodes:
        if node.is_leaf:
            continue
        brx = [child.branch[0][1] for child in node.children]
        bry = [child.branch[1][1] for child in node.children]
        links.append(((min(brx), max(brx)), (min(bry), max(bry))))

    # Plot edges
    for node in tree.nodes:
        node.branch_line = ax.plot(node.branch[0], node.branch[1], **kwargs)[0]

    # Plot lateral links between edges
    for link in links:
        ax.plot(link[0], link[1], **kwargs)

    xs = [node.x for node in tree.nodes] + [0]
    ys = [node.y for node in tree.nodes] + [0]

    wgs_analysis.plots.utils.set_xlim_filter_ticks(ax, min(xs), max(xs))
    wgs_analysis.plots.utils.set_ylim_filter_ticks(ax, min(ys), max(ys))

    ax.grid(False)

    if flip:
        if landscape:
            ax.invert_xaxis()
        else:
            ax.invert_yaxis()

    # Initialize all spines to invisible
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_tick_params(which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
    ax.yaxis.set_tick_params(which='both', left=False, right=False, labelleft=False, labelright=False)

    # Set spine and ticks to visible for x/y axis left/right/top/bottom
    def setup_axis(axis, spine):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_position(('outward', 10))
        axis.set_tick_params(which='both', **{spine:True, 'label'+spine:True})

    # Setup axis ticks and spine visibility based on orientation
    if landscape:
        if branch_length_attr is not None:
            setup_axis(ax.xaxis, 'bottom')
        if flip:
            setup_axis(ax.yaxis, 'left')
        else:
            setup_axis(ax.yaxis, 'right')
    else:
        if branch_length_attr is not None:
            setup_axis(ax.yaxis, 'left')
        if flip:
            setup_axis(ax.xaxis, 'bottom')
        else:
            setup_axis(ax.xaxis, 'top')

    # Shift plot limits to allow for full visibility of lines
    wgs_analysis.plots.utils.shift_lims(ax, 2, 2)

    # Trim spines to ticks for both axes
    wgs_analysis.plots.utils.trim_spines_to_ticks(ax)

    # Set leaf names if we have an attribute to query from leaf nodes
    leaf_ticks = list(range(len(list(tree.leaves))))
    if leaf_name_attr is not None:
        leaf_names = [getattr(leaf, leaf_name_attr) for leaf in tree.leaves]
        if landscape:
            ax.set_yticks(leaf_ticks)
            ax.set_yticklabels(leaf_names)
        else:
            ax.set_xticks(leaf_ticks)
            ax.set_xticklabels(leaf_names)


def plot_tree_nodes(ax, tree, node_color=None, node_label=False, node_attr=None, node_marker_size=6):
    """ Plot a tree.

    Args:
        ax (matplotlib.axes.Axes): plot axes
        tree (dollo.trees.TreeNode): tree to plot

    Kwargs:
        node_color (color or list of colors): single color for all nodes or color of each node by label
        node_label (bool): whether to draw node labels
        node_attr (str): node attribute to plot
        node_marker_size (int): size of circular node marker

    """

    for node in tree.nodes:

        if node_color is not None or node_label:

            if node_color is None:
                color = 'white'
            elif isinstance(node_color, basestring):
                color = node_color
            else:
                color =  node_color[node.label]

            ax.plot(node.x, node.y, 'bo', color=color, zorder=2, fillstyle='full', markersize=node_marker_size)

        node_text = None
        if node_label:
            node_text = str(node.label)
        elif node_attr is not None:
            node_text = str(getattr(node, node_attr))

        if node_text is not None:
            ax.text(node.x, node.y, node_text, ha='center', va='center', fontsize='small')


def plot_genotype_nodes(ax, tree, genotype_attr=None, genotype_colors=None, width=0.1, height=0.1):
    """ Plot a genotype tree.

    Args:
        ax (matplotlib.axes.Axes): plot axes
        tree (dollo.trees.TreeNode): tree to plot

    Kwargs:
        genotype_attr (str): attribute for presence absence vector on each node
        genotype_colors (list of colors): color for each variant in the genotype
        width (float): width of genotype boxes
        height (float): height of genotype boxes

    The tree must have an array of 0/1 indicators at each node denoting presence/absence
    of the variant for the genotype of each node.

    """

    for node in tree.nodes:

        offset_x = width * float(len(node.genotype)) / 2.
        offset_y = -height * 0.5

        for idx, present in enumerate(node.genotype):

            widget_x =node.x + width * idx - offset_x
            widget_y = node.y + offset_y

            color = 'white'

            if present == 1:
                if genotype_colors is None:
                    color = 'k'
                else:
                    color = genotype_colors[idx]

            ax.add_patch(Rectangle((widget_x, widget_y), width, height,
                facecolor=color, edgecolor='.75', linewidth=0.25, zorder=10))

        ax.add_patch(Rectangle((node.x - offset_x, node.y + offset_y), 2 * offset_x, height,
            facecolor=(0,0,0,0), edgecolor='k', linewidth=1., zorder=10))


def plot_branch_events(ax, tree, genotype_table, event_colors=None):
    """ Plot origin and loss on branches.

    Args:
        ax (matplotlib.axes.Axes): plot axes
        tree (dollo.trees.TreeNode): tree to plot
        genotype_table (pandas.DataFrame): table of genotypes per node

    KwArgs:
        event_colors (dict): event colors keyed by event id

    The genotype table should have node labels as row indices and event ids (ids of 
    snv/breakpoint clusters) as column labels.  Each 0/1 entry represents the
    presence or absence of the event for that node.

    """

    num_events = len(genotype_table.columns)

    if event_colors is None:
        if num_events > 8:
            palette = seaborn.color_palette('husl', num_events)
        else:
            palette = seaborn.color_palette('Set2', num_events)
        event_colors = dict(zip(genotype_table.columns.values, palette))

    for node in tree.nodes:

        markers = list()

        for event_id in genotype_table.columns.values:

            color = event_colors[event_id]

            node_present = genotype_table.loc[node.label, event_id]

            parent_present = 0
            if node.parent is not None:
                parent_present = genotype_table.loc[node.parent.label, event_id]

            if parent_present == 0 and node_present == 1:
                markers.append({'marker':'o', 'color':color, 'zorder':100, 'facecolor':'white', 's':50, 'lw':3})
            elif parent_present == 1 and node_present == 0:
                markers.append({'marker':'x', 'color':color, 'zorder':100, 's':50, 'lw':2})

        if len(markers) == 0:
            continue

        positions = np.linspace(0, 1, len(markers) + 2)[1:-1]

        for pos, marker in zip(positions, markers):
            x = pos * node.branch[0][0] + (1 - pos) * node.branch[0][1]
            y = pos * node.branch[1][0] + (1 - pos) * node.branch[1][1]
            ax.scatter(x, y, **marker)


def get_mutation_order(tree, nodes):
    """ Get mutation order placing ancestral first.

    Args:
        tree (dollo.trees.TreeNode): tree to plot
        nodes (pandas.DataFrame): nodes table

    Returns:
        list: event_ids from nodes table ordered by ancestrality

    """

    add_depths(tree, 0)
    add_positions(tree, itertools.count())

    for node in tree.nodes:
        nodes.loc[nodes['node'] == node.label, 'plot_depth'] = node.depth
        nodes.loc[nodes['node'] == node.label, 'plot_position'] = node.position

    event_origin = nodes[nodes['ml_origin'] == 1]
    event_origin = event_origin.sort(['plot_depth', 'plot_position'])
    event_order = event_origin['category'].values

    return event_order


def add_internal_node_leaves(tree):
    """ Add leaves to represent internal nodes.

    Args:
        tree (dollo.trees.TreeNode): tree to augment

    Returns:
        dollo.trees.TreeNode: tree with additional leaves

    Additional leaf nodes will have the same label as their internal node parent.

    """

    for node in list(tree.nodes):
        if not node.is_leaf:
            leaf = dollo.trees.TreeNode(None)
            leaf.label = node.label
            node.children.append(leaf)

    return tree



