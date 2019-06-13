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


def bins_to_segments(bins, sample_col, state_col, agg_values=None):
    """ Merge adjacent bins with the same state to produce a segment table.
    
    Args:
        bins (pandas.DataFrame): Binned copy number data
        sample_col (str): Sample column in bins
        state_col (str): State column to use to detect equality for merging adjacent bins
        agg_values (dict, optional): Additional values to aggregate and their agg. Defaults to None.
    
    Returns:
        pandas.DataFrame: Segments table of copy number data.
    """

    segments = []

    for sample, data in bins.groupby(sample_col):
        data = data.sort_values(['chr', 'start'])

        data['next_start'] = data['end'] + 1
        data['bins_adj'] = np.concatenate(
            [[None], data['start'].values[1:] == data['next_start'].values[:-1]])
        data['chr_adj'] = np.concatenate(
            [[None], data['chr'].values[1:] == data['chr'].values[:-1]])
        data['state_equal'] = np.concatenate(
            [[None], data[state_col].values[1:] == data[state_col].values[:-1]])
        data['adj_equal'] = (data['bins_adj'] & data['chr_adj'] & data['state_equal'])
        data['segment_idx'] = (~data['adj_equal']).cumsum()

        agg_cols = {
            'chr': 'first',
            'start': 'min',
            'end': 'max',
            state_col: 'first'}

        if agg_values is not None:
            agg_cols.update(agg_values)

        data = data.groupby('segment_idx').agg(agg_cols)
        data = data.reset_index(drop=True)
        data[sample_col] = sample

        segments.append(data)

    segments = pd.concat(segments)

    segments['chr'] = segments['chr'].astype(bins['chr'].dtype)

    return segments
