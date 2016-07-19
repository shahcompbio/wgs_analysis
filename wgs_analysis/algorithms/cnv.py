import numpy as np
import pandas as pd


def cluster_copies(copies):
    """
    Cluster copies by chromosome and copy state
    
    Args:
        copies (pandas.DataFrame): `cnv_site` table

    Returns:
        pandas.DataFrame: table of clustered copy numbers

    Cluster copies by copy state (major, minor, major_sub, minor_sub) and by
    chromosome.  Calculate the average of major_raw and minor_raw weighted
    by length.

    """
    
    copies_table = copies.copy()
    copies_table = copies_table.set_index(['chrom', 'start', 'end', 'site_id'])
    copies_table = copies_table[['major', 'minor', 'major_sub', 'minor_sub']]
    copies_table = copies_table.unstack().fillna(0.0)
    copies_table.columns = [':'.join(a) for a in copies_table.columns]
    
    copies_columns = list(copies_table.columns.values)
    
    copies_group = copies_table.reset_index().drop_duplicates(copies_columns)
    copies_group['copies_group'] = list(xrange(len(copies_group.index)))
    
    copies_table.reset_index(inplace=True)
    copies_table = copies_table.merge(copies_group[copies_columns+['copies_group']], left_on=copies_columns, right_on=copies_columns)
    
    copies = copies.merge(copies_table[['chrom', 'start', 'end', 'copies_group']], left_on=['chrom', 'start', 'end'], right_on=['chrom', 'start', 'end'])
    
    copies['major_raw'] = copies['major_raw'] * copies['length']
    copies['minor_raw'] = copies['minor_raw'] * copies['length']
    
    # Aggregate for grouping by copy group.
    # For each set of copies, aggregate by minimum which has no effect
    # because they are the same across the group
    agg_op = dict({'start':np.min, 'end':np.max, 'length':np.sum, 'major_raw':np.sum, 'minor_raw':np.sum})
    
    # Group and remove copy_group column
    copies = copies.groupby(['site_id', 'chrom', 'copies_group']).agg(agg_op).reset_index()
    
    copies['major_raw'] = copies['major_raw'] / copies['length']
    copies['minor_raw'] = copies['minor_raw'] / copies['length']
    
    return copies


