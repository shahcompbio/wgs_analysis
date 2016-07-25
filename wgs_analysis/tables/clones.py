import gzip
import pickle
import copy
import string
import numpy as np
import pandas as pd

import wgs_analysis.helpers as helpers
import wgs_analysis.plots as plots
import wgs_analysis.plots.trees
import wgs_analysis.algorithms as algorithms
import wgs_analysis.algorithms.tree


def load_wgss_node_prevalence(patient_id):

    node_prevalence = pd.read_csv(
        helpers.results_directory + '/clonal_phylogeny/clonal_prevalence_mcmc/wgss/clonal_prevalence/patient_{0}.tsv.gz'.format(patient_id),
        sep='\t', compression='gzip'
    )

    node_prevalence = node_prevalence.set_index(['clone_id', 'site_id'])['mean'].unstack()

    return node_prevalence


def load_wgss_genotypes(patient_id):

    genotypes = pd.read_csv(
        helpers.results_directory + '/clonal_phylogeny/clonal_prevalence_mcmc/wgss/nodes/patient_{0}.tsv.gz'.format(patient_id),
        sep='\t', compression='gzip'
    )

    genotypes.rename(columns={'event_id':'cluster_id'}, inplace=True)

    return genotypes


def load_wgss_tree(patient_id):

    tree_filename = helpers.results_directory + '/clonal_phylogeny/clonal_prevalence_mcmc/wgss/tree/patient_{0}.pickle.gz'.format(patient_id)

    with gzip.open(tree_filename, 'r') as f:
        tree = pickle.load(f)

    return tree


def load_sc_node_prevalence(patient_id):

    node_prevalence = pd.read_csv(
        helpers.results_directory + '/clonal_phylogeny/clonal_prevalence_mcmc/single_cell/clonal_prevalence/patient_{0}.tsv.gz'.format(patient_id),
        sep='\t', compression='gzip'
    )

    node_prevalence = node_prevalence.set_index(['clone_id', 'site_id'])['mean'].unstack()

    return node_prevalence


def load_sc_genotypes(patient_id):

    genotypes = pd.read_csv(
        helpers.results_directory + '/pyclone_single_cell_dollo/patient_{0}/nodes.tsv.gz'.format(patient_id),
        sep='\t', compression='gzip',
    )

    genotypes.rename(columns={'event_id':'cluster_id'}, inplace=True)

    return genotypes


def load_sc_tree(patient_id):

    tree = pickle.load(gzip.open(
        helpers.results_directory + '/clonal_phylogeny/clonal_prevalence_mcmc/single_cell/tree/patient_{0}.pickle.gz'.format(patient_id)))
    
    return tree


def get_wgss_pyclone_cluster_id(patient_id):
    data = helpers.read_table_data(patient_id, 'clonal_analysis/pyclone_wgss')
    data.set_index(['chrom', 'coord', 'ref', 'alt'], inplace=True)
    data = data.idxmax(axis=1).astype(int)
    data.name = 'cluster_id'
    data = data.reset_index()
    return data


def load_clone_data(patient_id):
    """ Load clone data for a patient.

    Args:
        patient_id (str): patient identifier

    Returns:
        dollo.trees.TreeNode: clone tree
        pandas.DataFrame: genotype table
        pandas.DataFrame: node prevalence table

    Load single cell data if available, otherwise wgss based clones.

    The leaves of the clone tree are the full set of clones.  All internal nodes
    have a descendent leaf with no mutations on the branch to that leaf.

    The genotype table will have node labels as row indices and event ids (ids of 
    snv/breakpoint clusters) as column labels.  Each 0/1 entry represents the
    presence or absence of the event for that node.  

    The node prevalence table table should have sample ids as row indices and leaf
    names as column labels.

    """

    # Load data
    if int(patient_id) in helpers.single_cell_patients:
        genotypes = load_sc_genotypes(patient_id)
        tree = load_sc_tree(patient_id)
        node_prevalence = load_sc_node_prevalence(patient_id)

    else:
        genotypes = load_wgss_genotypes(patient_id)
        tree = load_wgss_tree(patient_id)
        node_prevalence = load_wgss_node_prevalence(patient_id)

    # Filter normal clone, excluded samples
    site_names = helpers.get_patient_tumour_site_names(patient_id)
    node_prevalence = node_prevalence.loc[node_prevalence.index.astype(str) != 'normal', site_names]
    node_prevalence.index = node_prevalence.index.astype(int)

    # Patient 2 FFPE filtered
    if int(patient_id) == 2:
        wgs_site_names = helpers.get_wgss_patient_tumour_site_names(patient_id)
        node_prevalence = node_prevalence[wgs_site_names]

    # Renormalize
    node_prevalence = node_prevalence / node_prevalence.sum(axis=0)

    # Calculate tree cluster ids
    cluster_ids = set()
    for node in tree.nodes:
        cluster_ids.update(node.genotype)

    # Filter nodes table based on cluster ids
    genotypes = genotypes[genotypes['cluster_id'].isin(cluster_ids)]

    # Remove branches with no mutation
    tree = algorithms.tree.simplify_tree_by_genotype(tree, genotypes)

    # Add leaves representing internal nodes to the tree
    tree = plots.trees.add_internal_node_leaves(tree)

    # Letter names for leaves
    for leaf, name in zip(reversed(list(tree.leaves)), string.ascii_uppercase):
        leaf.name = name

    # Node prevalence table with columns as leaf names, row indices as site ids
    leaf_name = dict([(leaf.label, leaf.name) for leaf in tree.leaves])

    node_prevalence['leaf'] = pd.Series(leaf_name).reindex(node_prevalence.index)

    node_prevalence = (
        node_prevalence
        .set_index('leaf')
        .T
        .rename(index=lambda site_name: helpers.get_plot_id(patient_id, site_name))
    )

    # Genotype table, node labels as row indices, cluster ids as column names
    genotypes = genotypes.set_index(['node', 'cluster_id'])['ml_presence'].unstack()

    return tree, genotypes, node_prevalence


def calculate_cluster_origin(tree, genotypes):
    """ Calculate the origin node of each pyclone cluster.

    Args:
        tree (dollo.TreeNode): tree to simplify
        genotypes (pandas.DataFrame): presence absence table

    Returns:
        dict: clone (leaf name) keyed by cluster

    The presence absence table should have node labels as row indices and pyclone
    cluster id as column name.  Values are 0/1 for presence/absence.

    """

    cluster_presence = (
        genotypes
        .stack()
        .reset_index()
        .rename(columns={0:'presence'})
    )

    cluster_presence = (
        cluster_presence[cluster_presence['presence'] == 1]
        .drop(['presence'], axis=1)
    )

    for node in tree.nodes:
        node.clusters = set()

    for node_label, cluster_id in cluster_presence.values:
        for node in tree.nodes:
            if node.label == node_label:
                node.clusters.add(cluster_id)

    tree.add_parent()

    # Calculate the number of new SNVs that
    # distinguish parent from child
    for node in tree.nodes:
        node.origins = set()
        for cluster_id in node.clusters:
            if node.parent is None or cluster_id not in node.parent.clusters:
                node.origins.add(cluster_id)

    # Propogate to leaves representing internal nodes
    for node in tree.nodes:
        if node.parent is not None and node.parent.label == node.label:
            node.origins = node.parent.origins

    # Create a map between cluster id and leaf name
    cluster_origin = dict()
    for leaf in tree.leaves:
        for cluster_id in leaf.origins:
            cluster_origin[cluster_id] = leaf.name

    return cluster_origin


def calculate_num_gained(tree, genotypes, sizes):
    """ Count number of gained SNVs per clone.

    Args:
        tree (dollo.TreeNode): tree to simplify
        genotypes (pandas.DataFrame): presence absence table
        sizes (mapping): cluster sizes

    Returns:
        dict: number of gained SNVs keyed by clone (leaf name)

    The presence absence table should have node labels as row indices and pyclone
    cluster id as column name.  Values are 0/1 for presence/absence.

    The sizes should be a mapping type with pyclone cluster ids as keys and cluster
    sizes as values.

    > Does not count losses

    """

    cluster_origin = calculate_cluster_origin(tree, genotypes)

    num_gained = dict([(leaf.name, 0) for leaf in tree.leaves])
    for cluster_id, size in sizes.iteritems():
        if cluster_id in cluster_origin:
            num_gained[cluster_origin[cluster_id]] += size

    return num_gained


def load_prevalence_data(patient_id, clade=False):
    """ Load prevalences for a patient.

    Args:
        patient_id (int): patient

    KwArgs:
        clade (bool): clade prevalences

    Returns:
        pandas.DataFrame clone prevalence dataframe.

    Return dataframe has columns clone_id, site_id, num_gained_wgss, num_gained_deepseq, prevalence.

    """

    pyclone = pd.read_csv(
        helpers.results_directory + '/clonal_phylogeny/cluster_posteriors/patient_{0}.tsv.gz'.format(patient_id),
        sep='\t', compression='gzip'
    )
    
    pyclone_sizes = (
        pyclone
        .drop_duplicates(['cluster_id', 'size'])
        .set_index(['cluster_id'])['size']
    )

    wgss_sizes = helpers.get_pyclone_wgss_cluster_sizes(patient_id)

    tree, genotypes, node_prevalence = load_clone_data(patient_id)

    if clade:
        tree.add_parent()

        clade_prevalence = node_prevalence.copy() * 0
        node_clone = tree.get_attr_map('name')
        for node in tree.nodes:
            if node.parent is not None and node.parent.label == node.label:
                continue
            for leaf in node.leaves:
                clade_prevalence[node_clone[node.label]] += node_prevalence[leaf.name]

        prevalence = clade_prevalence
        
    else:
        prevalence = node_prevalence

    num_gained_wgss = calculate_num_gained(tree, genotypes, wgss_sizes)
    num_gained_deepseq = calculate_num_gained(tree, genotypes, pyclone_sizes)

    tissue_source = helpers.sample_info.loc[
        (helpers.sample_info['patient_id'] == int(patient_id)) &
        (helpers.sample_info['malignant'] == 'yes'),
        ['plot_id', 'tissue_source']
    ].rename(columns={'plot_id':'site_id'})

    anatomy_id = helpers.sample_info.loc[
        (helpers.sample_info['patient_id'] == int(patient_id)) &
        (helpers.sample_info['malignant'] == 'yes'),
        ['plot_id', 'anatomy_id']
    ].rename(columns={'plot_id':'site_id'})

    prevalence_table = (
        prevalence
        .stack()
        .reset_index()
        .rename(columns={0:'prevalence', 'leaf':'clone_id'})
        .merge(pd.DataFrame({'num_gained_wgss':num_gained_wgss}), left_on='clone_id', right_index=True)
        .merge(pd.DataFrame({'num_gained_deepseq':num_gained_deepseq}), left_on='clone_id', right_index=True)
        .merge(tissue_source, on='site_id', how='outer')
        .merge(anatomy_id, on='site_id', how='outer')
        .dropna()
    )

    return prevalence_table


def calculate_sample_classification(tree, node_prevalence, presence_threshold=0.01):
    """ Calculate branching classification for each sample.

    Args:
        tree (dollo.TreeNode): clone tree
        node_prevalence (pandas.DataFrame): node prevalence table

    KwArgs:
        presence_threshold (float): prevalence threshold for calling presence

    Returns:
        pandas.Series: classification with sample index

    Classify the branching relationship of clones present in each sample.

    """

    tree = copy.deepcopy(tree)
    clone_node = dict([(leaf.name, leaf.label) for leaf in tree.leaves])

    # Remove leaves representing internal nodes
    for node in list(tree.nodes):
        children = list()
        for child in node.children:
            if child.label != node.label:
                children.append(child)
        node.children = children

    def classify_sample_tree(prv):
        site_clones = prv[prv > presence_threshold].index.values
        site_labels = [clone_node[a] for a in site_clones]
        site_tree = algorithms.tree.simplify_tree(copy.deepcopy(tree), site_labels)
        return algorithms.tree.classify_tree(site_tree)

    sample_classification = node_prevalence.apply(classify_sample_tree, axis=1)

    return sample_classification



def load_sample_composition_data(patient_id, presence_threshold=0.01):
    """ Calculate a table of clonal composition statistics per sample.

    Args:
        patient_id (str): patient
        presence_threshold (float): minimum prevalence for clonal presence

    Returns:
        pandas.DataFrame: table of per sample info with columns
          'site_id', 'divergence', 'entropy', 'sample_class'

    'sample_class' is one of 'chain', 'branched', 'pure'

    """

    pyclone = pd.read_csv(
        helpers.results_directory + '/clonal_phylogeny/cluster_posteriors/patient_{0}.tsv.gz'.format(patient_id),
        sep='\t', compression='gzip'
    )
    
    pyclone_sizes = (
        pyclone
        .drop_duplicates(['cluster_id', 'size'])
        .set_index(['cluster_id'])['size']
    )

    wgss_sizes = helpers.get_pyclone_wgss_cluster_sizes(patient_id)

    tree, genotypes, node_prevalence = load_clone_data(patient_id)

    data = node_prevalence.stack().reset_index().rename(columns={0:'prevalence', 'leaf':'clone'})
    data = data[data['prevalence'] > presence_threshold]

    label_clone = tree.get_attr_map('name')

    # Create table of 'cluster_id', 'present', 'size' indexed by clone
    clone_genotypes = genotypes.reindex(label_clone.keys()).rename(index=label_clone)
    clone_genotypes.index.name = 'clone'
    clone_genotypes = clone_genotypes.stack().reset_index().rename(columns={0:'present'})
    clone_genotypes = clone_genotypes[clone_genotypes['present'] > 0]
    clone_genotypes = clone_genotypes.set_index('clone').sort_index()

    site_stats = list()

    for site_id, site_data in data.groupby('site_id'):
        clones = set(site_data['clone'].values)
        union = set()
        intersection = set(clone_genotypes['cluster_id'].unique())
        for clone in clones:
            clusters = clone_genotypes.loc[clone:clone, 'cluster_id'].values
            union = union.union(clusters)
            intersection = intersection.intersection(clusters)
        union_size = sum([wgss_sizes[a] for a in union])
        intersection_size = sum([wgss_sizes[a] for a in intersection])
        divergence = 1.0 - float(intersection_size) / float(union_size)
        entropy = -np.sum(site_data['prevalence'] * np.log(site_data['prevalence']))

        site_stats.append((site_id, divergence, entropy))

    sample_data = pd.DataFrame(site_stats, columns=['site_id', 'divergence', 'entropy'])

    sample_data.set_index('site_id', inplace=True)
    sample_data['sample_class'] = calculate_sample_classification(tree, node_prevalence)
    sample_data.reset_index(inplace=True)

    return sample_data


