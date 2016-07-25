'''
# Tree related tables

Generation of derivative tables for tree related questions.

'''

import numpy as np
import pandas as pd
import scipy
import dollo.run
import scipy.misc

import wgs_analysis.helpers as helpers
import wgs_analysis.params as params
import wgs_analysis.annotation as annotation
import wgs_analysis.annotation.cnv
import wgs_analysis.algorithms as algorithms
import wgs_analysis.algorithms.tree
import wgs_analysis.plots as plots
import wgs_analysis.plots.utils


def convergent_loh(tree, cnv):
    """ Calculate a table of loh that doesnt fit a tree.

    Args:
        tree (TreeNode): tree to test cnvs for convergence
        cnv (pandas.DataFrame): table of cnvs

    Returns:
        pandas.DataFrame: table of loh events with columns `length`, `is_loh`, `is_concordant`, `node_list`

    """

    cnv['length'] = cnv['end'] - cnv['start']
    cnv['length'] = cnv['length'].astype(float)

    cnv = annotation.cnv.annotate_cnv_type(cnv)

    loh_site_matrix = cnv.set_index(['chrom', 'start', 'end', 'length', 'site_id'])['loh'].unstack().dropna()

    def add_loh_state(tree, loh_sites):
        for child in tree.children:
            add_loh_state(child, loh_sites)
        if tree.is_leaf:
            tree.loh = (tree.name in loh_sites) * 1
        else:
            tree.loh = (all([child.loh for child in tree.children])) * 1

    def add_loh_origin(tree):
        if tree.loh:
            tree.loh_origin = 1
            for descendent in tree.descendents:
                descendent.loh_origin = 0
        else:
            tree.loh_origin = 0
            for child in tree.children:
                add_loh_origin(child)

    def calc_loh_origins(row):
        loh_sites = set(row[row > 0].index.values)
        add_loh_state(tree, loh_sites)
        add_loh_origin(tree)
        loh_nodes = pd.Series(tree.get_attr_map('loh_origin'))
        return loh_nodes
        
    loh_origin_matrix = loh_site_matrix.apply(calc_loh_origins, axis=1)

    nodes = list(loh_origin_matrix.columns.values)

    loh_origin_matrix = loh_origin_matrix.reset_index()

    loh_origin_matrix = loh_origin_matrix.set_index(nodes)

    loh_origin_lengths = loh_origin_matrix.groupby(level=range(len(nodes)))['length'].sum().reset_index()

    def format_node_list(row):
        return ','.join([str(a) for a in row[row > 0].index])

    loh_origin_lengths['is_loh'] = (loh_origin_lengths[nodes].sum(axis=1) >= 1) * 1
    loh_origin_lengths['is_concordant'] = (loh_origin_lengths[nodes].sum(axis=1) <= 1) * 1
    loh_origin_lengths['node_list'] = loh_origin_lengths[nodes].apply(format_node_list, axis=1)
    loh_origin_lengths = loh_origin_lengths[loh_origin_lengths['is_loh'] == 1]
    loh_origin_lengths = loh_origin_lengths.drop(nodes, axis=1)

    return loh_origin_lengths


def snvs_deleted_table(nodes):

    nodes['is_lost'] = nodes['ml_loss']
    nodes['is_deleted'] = (nodes['deletion'] == 1) & (nodes['is_lost'] == 1)

    data = nodes.groupby(['chrom', 'coord', 'ref', 'alt'])[['is_lost', 'is_deleted']]\
                .any()\
                .reset_index()

    data = data[data['is_lost']].groupby(['is_deleted'])\
                                .size()\
                                .reset_index()\
                                .rename(columns={0:'count'})

    data['is_deleted'] *= 1

    return data


def snvs_possibly_deleted_table():

    lost_table = list()

    for patient_id in helpers.get_wgss_patient_ids():

        tree = helpers.get_snv_tree(patient_id)
        nodes = helpers.get_snv_nodes(patient_id)

        siblings = list()
        for node in tree.nodes:
            if node.is_leaf:
                continue
            assert len(node.children) == 2
            child_labels = [child.label for child in node.children]
            siblings.append(tuple(child_labels))
            siblings.append(tuple(reversed(child_labels)))
        siblings = pd.DataFrame(siblings, columns=['node', 'sibling_node'])

        sibling_nodes = nodes[['chrom', 'coord', 'ref', 'alt', 'node', 'deletion']].copy()
        sibling_nodes = sibling_nodes.rename(columns={'node':'sibling_node', 'deletion':'sibling_deletion'})

        nodes = nodes.merge(siblings)

        nodes = nodes.merge(sibling_nodes)

        loss = nodes.loc[(nodes['ml_origin'] == 1) &
                         (nodes['sibling_deletion'] == 1),
                         ['chrom', 'coord', 'ref', 'alt', 'node', 'sibling_node']]

        lost_table.append((patient_id, len(loss.index)))

    lost_table = pd.DataFrame(lost_table, columns=('patient_id', 'loss_count'))

    return lost_table


def correlate_patient_origin_loss():

    event_table = list()

    for patient_id in helpers.get_wgss_patient_ids():

        snv_tree = helpers.get_snv_tree(patient_id)
        breakpoint_tree = helpers.get_breakpoint_tree(patient_id)

        try:
            snv_breakpoint_nodes = algorithms.tree.align_trees(snv_tree, breakpoint_tree)
        except ValueError:
            print 'patient {0} has different breakpoint and SNV trees'.format(patient_id)
            continue

        snv_breakpoint_nodes = pd.DataFrame(snv_breakpoint_nodes, columns=['snv_node', 'breakpoint_node'])

        snv_nodes = helpers.get_snv_nodes(patient_id)
        snv_origin_counts = snv_nodes.groupby('node')['ml_origin'].sum().reset_index().rename(columns={'ml_origin':'count'})
        snv_loss_counts = snv_nodes.groupby('node')['ml_loss'].sum().reset_index().rename(columns={'ml_loss':'count'})

        snv_counts = pd.merge(snv_origin_counts, snv_loss_counts, on='node', suffixes=('_origin', '_loss'))

        breakpoint_nodes = helpers.get_breakpoint_nodes(patient_id)
        breakpoint_origin_counts = breakpoint_nodes.groupby('node')['ml_origin'].sum().reset_index().rename(columns={'ml_origin':'count'})
        breakpoint_loss_counts = breakpoint_nodes.groupby('node')['ml_loss'].sum().reset_index().rename(columns={'ml_loss':'count'})

        breakpoint_counts = pd.merge(breakpoint_origin_counts, breakpoint_loss_counts, on='node', suffixes=('_origin', '_loss'))

        event_counts = snv_breakpoint_nodes.copy()

        event_counts = event_counts.merge(snv_counts, left_on='snv_node', right_on='node')\
                                   .drop(['node'], axis=1)\
                                   .rename(columns={'count_origin':'count_origin_snv',
                                                    'count_loss':'count_loss_snv'})

        event_counts = event_counts.merge(breakpoint_counts, left_on='breakpoint_node', right_on='node')\
                                   .drop(['node'], axis=1)\
                                   .rename(columns={'count_origin':'count_origin_breakpoint',
                                                    'count_loss':'count_loss_breakpoint'})

        event_counts['patient_id'] = patient_id

        event_table.append(event_counts)

    event_table = pd.concat(event_table, ignore_index=True)

    return event_table

    
class gaussian_kde_bw(scipy.stats.gaussian_kde):
    def __init__(self, dataset, bw):
        self.covariance = bw**2.
        scipy.stats.gaussian_kde.__init__(self, dataset)
    def _compute_covariance(self):
        self.inv_cov = 1.0 / self.covariance
        self._norm_factor = np.sqrt(2*np.pi*self.covariance) * self.n


def snvs_loss_density_table(bw=1e8):

    density_table = list()

    for patient_id in helpers.get_wgss_patient_ids():

        nodes = helpers.get_snv_nodes(patient_id)

        lost = nodes[nodes['ml_loss'] == 1]
        lost = lost.drop_duplicates(['chrom', 'coord', 'ref', 'alt'])

        lost.set_index('chrom', inplace=True)
        lost['chromosome_start'] = plots.utils.chromosome_start
        lost.reset_index(inplace=True)

        lost['coord'] = lost['coord'] + lost['chromosome_start']

        density = gaussian_kde_bw(lost['coord'].values, bw)

        density_eval = lost['coord'].values
        density_eval = np.linspace(0, 3e9, 1000000)

        mean_density = density(density_eval).mean()
        count = len(lost['coord'].values)

        density_table.append((patient_id, mean_density, count))

    density_table = pd.DataFrame(density_table, columns=('patient_id', 'density', 'count'))

    return density_table


def calculate_presence_absence_genotypes(patient_id, absence_threshold=0.5):
    """ Nodes table for pyclone clusters based on presence absence.

    Create a table of maximum likelihood indicators for origin presence and
    loss at each node in the wgss tree.  Events are pyclone clusters and 
    presence at leaves is determined based on the cluster posteriors for 0
    and 1 prevalence.
    
    """

    wgss_site_ids = helpers.get_wgss_patient_tumour_site_names(patient_id)

    tree = helpers.get_snv_tree(patient_id)

    pyclone = pd.read_csv(
        helpers.results_directory + '/clonal_phylogeny/cluster_posteriors/patient_{0}.tsv.gz'.format(patient_id),
        sep='\t', compression='gzip'
    )

    pyclone.rename(columns={'cluster_id':'cluster_id', 'sample_id':'site_id'}, inplace=True)
    pyclone = pyclone[pyclone['site_id'].isin(wgss_site_ids)]
    pyclone.drop('size', axis=1, inplace=True)
    pyclone.set_index(['cluster_id', 'site_id'], inplace=True)

    def compute_posterior_probability(log_pdf, a, b):
        step_size = log_pdf.index[1] - log_pdf.index[0]
        log_step_size = pd.np.log(step_size)
        x = log_pdf.index[(log_pdf.index >= a) & (log_pdf.index <= b)]
        log_p = scipy.misc.logsumexp(log_step_size + log_pdf.loc[x])
        return log_p

    pyclone.columns = pyclone.columns.astype(float)
    
    ll_absent = pyclone.apply(compute_posterior_probability, axis=1, args=(0, absence_threshold))
    ll_present = pyclone.apply(compute_posterior_probability, axis=1, args=(absence_threshold, 1 ))
    
    ll = pd.concat([ll_absent, ll_present], axis=1)
    ll.rename(columns={0 : 'log_likelihood_absent', 1 : 'log_likelihood_present'}, inplace=True)
    
    ll = ll.T.stack().T

    tree.add_parent()
    tree.add_ancestors()

    ml_data = dollo.run.compute_max_likelihood_indicators(ll, tree, 0.5)

    nodes_table = pd.DataFrame(ml_data)
    nodes_table.reset_index(inplace=True)

    return nodes_table


