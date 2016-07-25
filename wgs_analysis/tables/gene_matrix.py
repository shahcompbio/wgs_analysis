'''
# Gene matrix table generation

Functions for generating tables of mutations/rearrangements for use with gene matrix plots and selection of interesting
candidates.

'''

import pandas as pd
import numpy as np

import wgs_analysis.helpers as helpers
import wgs_analysis.tables as tables


def sort_matrix(matrix):
    matrix['sort_value'] = matrix.apply(lambda x: get_sort_value(x), axis=1)
    
    matrix = matrix.sort(columns='sort_value', ascending=False)
    
    matrix = matrix.drop('sort_value', axis=1)
    
    return matrix


def get_sort_value(row):
    binary_str = ''
    
    for x in row:
        if np.isnan(x):
            x = 1
        
        binary_str += str(int(x))
    
    return int(binary_str, 2)


def _build_gene_matrix(data):

    gene_matrix = data.set_index(['event_name', 'gene_name', 'site_id'])['present'].unstack(level=-1)

    gene_matrix = gene_matrix.reset_index().drop('event_name', axis=1)

    gene_matrix = gene_matrix.drop_duplicates()

    gene_matrix = gene_matrix.set_index('gene_name')

    gene_matrix = sort_matrix(gene_matrix)
    
    gene_matrix = sort_matrix(gene_matrix.T).T
    
    gene_matrix = gene_matrix.rename(columns=lambda a: helpers.plot_ids[a])
    
    return gene_matrix


def _is_snv_present(row, min_depth):
    depth = row['ref_counts'] + row['alt_counts']
    
    if depth < min_depth:
        return np.nan
    
    elif row['alt_counts'] > 0:
        return 1
    
    else:
        return 0


def _is_indel_present(row):
    if row['alt_counts'] > 0:
        return 1
    
    else:
        return 0


def _is_breakpoint_present(row):
    if np.isnan(row['num_span']):
        return np.nan
    
    elif row['num_span'] > 0:
        return 1
    
    else:
        return 0
    

def load_filtered_snvs(patient_id):
    data = helpers.get_snvs(patient_id)

    data = data[data['site_id'].isin(helpers.get_patient_tumour_site_names(patient_id))]

    data = data[data.snpeff_impact.isin(['HIGH', 'MODERATE'])]

    data = data[data.gene_name.isin(helpers.target_snv_genes)]

    return data


def load_snv_matrix(patient_id, min_present_depth=10):
    data = load_filtered_snvs(patient_id)

    if len(data.index) == 0:
        return pd.DataFrame()

    def event_name(row):
        return ':'.join(['snv', str(row['chrom']), str(row['coord'])])
    
    data['event_name'] = data.apply(event_name, axis=1)
    
    data['present'] = data.apply(lambda x : _is_snv_present(x, min_present_depth), axis=1)
  
    data = data[['site_id', 'event_name', 'gene_name', 'present']].drop_duplicates()

    if len(data.index) == 0:
        return pd.DataFrame()

    return _build_gene_matrix(data)


def load_filtered_breakpoints(patient_id):
    data = helpers.get_breakpoints(patient_id)

    expression = helpers.get_recentred_expression(patient_id)

    data = tables.rearrangement.expression_correlated(data, expression)

    data = data[data.gene_name.isin(helpers.target_genes)]

    return data


def load_breakpoint_matrix(patient_id):
    data = load_filtered_breakpoints(patient_id)

    def event_name(row):
        return ':'.join(['brk', str(row['cluster_id']), str(row['cluster_end'])])
    
    data['event_name'] = data.apply(event_name, axis=1)
    
    data['present'] = data.apply(_is_breakpoint_present, axis=1) 

    data = data[['site_id', 'event_name', 'gene_name', 'present']].drop_duplicates()
    
    if len(data.index) == 0:
        return pd.DataFrame()    
    
    return _build_gene_matrix(data)


def load_filtered_indels(patient_id):
    data = helpers.get_indels(patient_id)

    data = data[data['site_id'].isin(helpers.get_patient_tumour_site_names(patient_id))]
    
    data = data[data.gene_name.isin(helpers.target_genes)]

    data.set_index(['chrom', 'coord', 'ref', 'alt'], inplace=True)
    data['max_alt_counts'] = data.set_index('site_id', append=True)['alt_counts'].unstack().max(axis=1)
    data.reset_index(inplace=True)

    data = data[data['max_alt_counts'] > 0]

    return data


def load_indel_matrix(patient_id, min_present_depth=10):
    data = load_filtered_indels(patient_id)

    if len(data.index) == 0:
        return pd.DataFrame()    

    index_columns = ['chrom', 'coord', 'ref', 'alt', 'gene_name']

    data = data.set_index(index_columns + ['site_id'])[['ref_counts', 'alt_counts']].unstack().fillna(0).stack().reset_index()

    def event_name(row):
        return ':'.join(['indel', str(row['chrom']), str(row['coord'])])
    
    data['event_name'] = data.apply(event_name, axis=1)
    
    data['present'] = data.apply(_is_indel_present, axis=1)

    data = data[['site_id', 'event_name', 'gene_name', 'present']].drop_duplicates()
    
    if len(data.index) == 0:
        return pd.DataFrame()    
    
    return _build_gene_matrix(data)


def load_cn_matrix(patient_id, cnv_type):

    if cnv_type not in ('amp', 'hdel'):
        raise ValueError(cnv_type)

    gene_regions = helpers.load_gene_regions()

    if cnv_type == 'amp':
        target_genes = helpers.target_amp_genes
    elif cnv_type == 'hdel':
        target_genes = helpers.target_del_genes

    target_gene_regions = gene_regions[gene_regions['gene_name'].isin(target_genes)]

    copies = helpers.get_copies(patient_id)

    stats = helpers.get_copies_stats(patient_id)

    site_ids = list(stats['site_id'].unique())

    stats['ploidy'] = (stats['minor_size'] + stats['major_size']) / 3e9

    copies = copies.merge(stats[['site_id', 'ploidy']], left_on='site_id', right_on='site_id')

    copies = copies.rename(columns={'site_id':'site_id'})

    data = list()

    for _, row in target_gene_regions.iterrows():

        ovlap = copies[(copies['chrom'] == row['chrom']) &
                       (copies['end'] >= row['start']) &
                       (copies['start'] <= row['end'])]

        if len(ovlap.index) == 0:
            continue

        if cnv_type == 'amp':
            variant = ovlap.loc[ovlap['major'] > 3. * ovlap['ploidy'], ['site_id']].drop_duplicates()
        elif cnv_type == 'hdel':
            variant = ovlap.loc[ovlap['major'] == 0, ['site_id']].drop_duplicates()

        if len(variant.index) == 0:
            continue

        variant['gene_name'] = row['gene_name']

        data.append(variant)

    if len(data) == 0:
        return pd.DataFrame()

    data = pd.concat(data, ignore_index=True)

    data['present'] = 1

    data = data.set_index(['gene_name', 'site_id'])['present']\
               .unstack()\
               .reindex(columns=site_ids)\
               .fillna(0)\
               .stack()\
               .reset_index()
    data.columns = ['gene_name', 'site_id', 'present']

    data['event_name'] = data['gene_name'] + '_' + cnv_type

    if len(data.index) == 0:
        return pd.DataFrame()    
    
    return _build_gene_matrix(data)

#=======================================================================================================================
# Deepseq
#=======================================================================================================================
def load_filtered_deeseq_snvs(patient_id):
    data = helpers.get_deepseq_snvs(patient_id)

    data = data[data['site_id'].isin(helpers.get_patient_tumour_site_names(patient_id))]

    data = data[data.snpeff_impact.isin(['HIGH', 'MODERATE'])]
    
    data = data[~data.primer_id.str.contains('amplicrazy')]

    return data

def load_deepseq_snv_matrix(patient_id, p_value_threshold=1e-6):
    data = load_filtered_deeseq_snvs(patient_id)

    if len(data.index) == 0:
        return pd.DataFrame()

    def event_name(row):
        return ':'.join(['snv', str(row['chrom']), str(row['coord'])])
    
    data['event_name'] = data.apply(event_name, axis=1)
    
    data['present'] = data.apply(lambda x: _is_deeqseq_snv_present(x, p_value_threshold), axis=1)
  
    data = data[['site_id', 'event_name', 'gene_name', 'present']].drop_duplicates()

    if len(data.index) == 0:
        return pd.DataFrame()

    return sort_matrix(_build_gene_matrix(data))

def _is_deeqseq_snv_present(row, threshold):
    return row['alt_p_value'] < threshold

def load_filtered_deeqpseq_breakpoints(patient_id, min_counts=5):
    data = helpers.get_deepseq_breakpoints(patient_id)

    germline = data.loc[
        (data['site_id'] == 'normal_blood') &
        (data['count'] >= min_counts),
        'seq_id'].unique()

    data = data[~data['seq_id'].isin(germline)]
    
    data = data[data['site_id'].isin(helpers.get_patient_tumour_site_names(patient_id))]
    
    return data

def load_deepseq_breakpoint_matrix(patient_id, min_counts=5):
    data = load_filtered_deeqpseq_breakpoints(patient_id, min_counts=min_counts)

    if len(data.index) == 0:
        return pd.DataFrame()

    def event_name(row):
        return row['seq_id']

    data['event_name'] = data.apply(event_name, axis=1)

    data['gene_name'] = (
        data['gene_name'] + ' ' +
        data['chrom_1'] + data['strand_1'] + data['coord_1'].astype(str) + ':' + 
        data['chrom_2'] + data['strand_2'] + data['coord_2'].astype(str))

    data['present'] = data.apply(lambda x: _is_deepseq_breakpoint_present(x, min_counts), axis=1)
  
    data = data[['site_id', 'event_name', 'gene_name', 'present']].drop_duplicates()

    if len(data.index) == 0:
        return pd.DataFrame()

    return sort_matrix(_build_gene_matrix(data))

def _is_deepseq_breakpoint_present(row, threshold):
    return row['count'] >= threshold