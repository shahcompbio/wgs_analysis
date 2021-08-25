'''
CNV annotation functionality

Annotations for cnv data.

Functions take pandas tables and add additional columns.
'''

import collections
import numpy as np
import pandas as pd

import wgs_analysis.algorithms as algorithms
import wgs_analysis.algorithms.merge


def label_presence(node, presence):
    """ Add present label to nodes

    Args:
        node (wgs_analysis.trees.TreeNode): tree node
        presence (dict): dictionary of presences by leaf name

    Add the present label to nodes.  By default nodes are labeled absent.  If any 
    of a nodes children have been labeled present, label that node as present.

    """

    if node.is_leaf:

        node.present = presence[node.name]

    else:

        node.present = 0

        for child in node.children:

            label_presence(child, presence)

            if child.present:
                node.present = 1


class InconsistentError(Exception):
    pass

def check_consistency(node):
    """ Check presence consistency

    Args:
        node (wgs_analysis.trees.TreeNode): tree node

    Raises:
        InconsistentError

    Check that if a node is labeled as present, so are all its children,
    otherwise raise an exception.

    """
    for child in node.children:

        if node.present and not child.present:
            raise InconsistentError()

        check_consistency(child)


def annotate_presence_timing(variants, tree, column_name, presence_f):
    """ Annotate timing of a presence absence feature

    Args:
        cnv (pandas.DataFrame): variant_site table
        tree (wgs_analysis.trees.TreeNode): tree
        column_name (str): column name to annotate
        presence_f (callable): function to calculate a dictionary of presences by site

    Returns:
        pandas.DataFrame: original table with additional column

    Annotate a variant table with the inferred timing of a presence absence event.

    """

    def calculate_timing(df):

        df = df.set_index('site_id')

        is_present = presence_f(df).to_dict()

        label_presence(tree, is_present)

        try:
            check_consistency(tree)
        except InconsistentError:
            return 'inconsistent'

        if np.sum(is_present) == len(df.index):
            return 'ancestral'
        elif np.sum(is_present) == 1:
            return 'private'
        else:
            return 'descendent'

    variants.set_index(['chrom', 'coord'], inplace=True)
    variants[column_name] = variants.groupby(level=[0, 1]).apply(calculate_timing)
    variants.reset_index(inplace=True)

    return variants


def annotate_loh_timing(variants, tree):
    """ Add loh_timing column

    Args:
        cnv (pandas.DataFrame): variant_site table
        tree (wgs_analysis.trees.TreeNode): tree

    Returns:
        pandas.DataFrame: original table with `loh_timing` column

    Annotate a variant table with the inferred timing of LOH events.

    """

    def calculate_loh_presence(df):
        return (df.minor == 0) * 1

    return annotate_presence_timing(variants, tree, 'loh_timing', calculate_loh_presence)


def annotate_hdel_timing(variants, tree):
    """ Add hdel_timing column

    Args:
        cnv (pandas.DataFrame): variant_site table
        tree (wgs_analysis.trees.TreeNode): tree

    Returns:
        pandas.DataFrame: original table with `hdel_timing` column

    Annotate a variant table with the inferred timing of homozygous deletion events.

    """

    def calculate_loh_presence(df):
        return (df.minor == 0) * 1

    return annotate_presence_timing(variants, tree, 'hdel_timing', calculate_loh_presence)


def annotate_cnv_type(copies):
    """ Add various cnv type columns

    Args:
        copies (pandas.DataFrame): cnv table

    Returns:
        pandas.DataFrame: original table with additional column
    """

    # LOH status for each segment
    copies['loh'] = (copies['minor'] < 0.5) * 1

    # Homozygous deletion status for each segment
    copies['hdel'] = (copies['major'] < 0.5) * 1

    # Subclonal LOH status (only if not loh)
    copies['subclonal_loh'] = (np.minimum(copies['minor'] + copies['minor_sub'], copies['major'] + copies['major_sub']) < 0.5) * 1
    copies['subclonal_loh'] = copies['subclonal_loh'] * (1 - copies['loh'])

    # Subclonal homozygous deletion status (only if not dominant hdel)
    copies['subclonal_hdel'] = (np.maximum(copies['minor'] + copies['minor_sub'], copies['major'] + copies['major_sub']) < 0.5) * 1
    copies['subclonal_hdel'] = copies['subclonal_hdel'] * (1 - copies['hdel'])

    return copies


def annotate_position_cnv(positional, cnv):
    """ Add cnv columns to positional data

    Args:
        positional (pandas.DataFrame): positions table with 'site_id', 'chrom', 'coord' columns
        cnv (pandas.DataFrame): cnv data with 'site_id', 'chrom', 'start', 'end' and additional copies columns

    Returns:
        pandas.DataFrame: positions table with additional columns from cnv table
    """
    merged = algorithms.merge.position_segment_merge(positional, cnv)

    cnv = cnv.merge(merged, on=['chrom', 'start', 'end']).drop(['start', 'end'], axis=1)

    positional = positional.merge(cnv, on=['chrom', 'coord'], how='left')

    return positional







