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


def aggregate_adjacent(cnv, value_cols=(), stable_cols=(), length_normalized_cols=(), summed_cols=()):
    """ Aggregate adjacent segments with similar copy number state.

    see: https://github.com/amcpherson/remixt/blob/master/remixt/segalg.py

    Args:
        cnv (pandas.DataFrame): copy number table

    KwArgs:
        value_cols (list): list of columns to compare for equivalent copy number state
        stable_cols (list): columns for which values are the same between equivalent states
        length_normalized_cols (list): columns that are width normalized for equivalent states

    Returns:
        pandas.DataFrame: copy number with adjacent segments aggregated
    """

    # Group segments with same state
    cnv = cnv.sort_values(['chromosome', 'start'])
    cnv['chromosome_index'] = np.searchsorted(np.unique(cnv['chromosome']), cnv['chromosome'])
    cnv['diff'] = cnv[['chromosome_index'] + value_cols].diff().abs().sum(axis=1)
    cnv['is_diff'] = (cnv['diff'] != 0)
    cnv['cn_group'] = cnv['is_diff'].cumsum()

    def agg_segments(df):
        a = df[stable_cols].iloc[0]

        a['chromosome'] = df['chromosome'].min()
        a['start'] = df['start'].min()
        a['end'] = df['end'].max()
        a['length'] = df['length'].sum()

        for col in length_normalized_cols:
            a[col] = (df[col] * df['length']).sum() / (df['length'].sum() + 1e-16)

        for col in summed_cols:
            a[col] = df[col].sum()

        return a

    aggregated = cnv.groupby('cn_group').apply(agg_segments)

    for col in aggregated:
        aggregated[col] = aggregated[col].astype(cnv[col].dtype)

    return aggregated


def calculate_gene_copy(cnv, genes, cols):
    """ Calculate the copy number segments overlapping each gene

    Args:
        cnv (pandas.DataFrame): copy number table
        genes (pandas.DataFrame): gene table
        cols (list of str): copy number columns

    Returns:
        pandas.DataFrame: segment copy number for each gene

    The input copy number table is assumed to have columns: 
        'chromosome', 'start', 'end', 'width'
    in addition to the columns from cols

    The input genes table is assumed to have columns:
        'chromosome', 'gene_start', 'gene_end', 'gene_name',
        'gene_id'

    The output segment copy number table should have columns:
        'chromosome', 'gene_start', 'gene_end', 'gene_name',
        'gene_id', 'overlap_start', 'overlap_end', 'overlap_width',
    in addition to the columns from cols

    """
    data = []

    for chromosome in cnv['chromosome'].unique():
        chr_cnv = cnv[cnv['chromosome'] == chromosome]
        chr_genes = genes[genes['chromosome'] == chromosome]
        
        # Iterate through segments, calculate overlapping genes
        for idx, row in chr_cnv.iterrows():

            # Subset overlapping genes
            overlapping_genes = chr_genes[
                ~((chr_genes['gene_end'] < row['start']) | (chr_genes['gene_start'] > row['end']))].copy()

            overlapping_genes['overlap_start'] = overlapping_genes['gene_start'].clip(lower=row['start'])
            overlapping_genes['overlap_end'] = overlapping_genes['gene_end'].clip(upper=row['end'])
            overlapping_genes['overlap_width'] = overlapping_genes['overlap_end'] - overlapping_genes['overlap_start'] - 1

            for col in cols:
                overlapping_genes[col] = row[col]

            data.append(overlapping_genes)

    data = pd.concat(data, ignore_index=True)

    return data

