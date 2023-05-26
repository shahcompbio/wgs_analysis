import os
import seaborn
import lda
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfaidx
import matplotlib.font_manager
import pkg_resources


def reverse_complement(sequence):
    return sequence[::-1].translate(str.maketrans('ACTGactg','TGACtgac'))


def complement(sequence):
    return sequence.translate(str.maketrans('ACTGactg','TGACtgac'))


def fit_sample_signatures(snvs_table, sig_prob, subset_col):

    # Filter samples with fewer than 100 SNVs
    snv_counts = snvs_table.groupby(subset_col).size()
    snvs_table = snvs_table.set_index(subset_col).loc[snv_counts[snv_counts > 100].index].reset_index()

    tri_nuc_table = (snvs_table.groupby([subset_col, 'tri_nuc_idx'])
        .size().unstack().fillna(0).astype(int))

    if len(tri_nuc_table.index) == 0:
        return pd.DataFrame()

    model = lda.LDA(n_topics=len(sig_prob.columns), random_state=0, n_iter=10000, alpha=0.01)
    model.components_ = sig_prob.values.T

    sample_sig = model.transform(tri_nuc_table.values, max_iter=1000)

    sample_sig = pd.DataFrame(sample_sig, index=tri_nuc_table.index, columns=sig_prob.columns)
    sample_sig.index.name = 'Sample'
    sample_sig.columns = [a[len('Signature '):] for a in sample_sig.columns]
    sample_sig.columns.name = 'Signature'

    return sample_sig


def plot_signature_heatmap(sample_sig):
    if sample_sig.shape[0] <= 1:
        return plt.figure(figsize=(8,5))
    g = seaborn.clustermap(sample_sig, figsize=(8,5))
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    return g.fig


def test_ancestral_descendant(data):
    data = dict(list(data.groupby(by=lambda a: a.endswith('Node0'))))
    return scipy.stats.mannwhitneyu(data[True], data[False])[1]


def plot_signature_boxplots(sample_sig, pvalue_threshold=0.01):

    # Show only signatures with p-value less than specified
    test_pvalue = sample_sig.apply(test_ancestral_descendant)
    data = sample_sig.loc[:,test_pvalue[test_pvalue < pvalue_threshold].index]

    data = data.stack()
    data.name = 'Proportion'
    data = data.reset_index()
    data['is_node0'] = data['Sample'].apply(lambda a: a.endswith('Node0'))
    data['Branch'] = data['is_node0'].apply(lambda a: ('Descendant', 'Ancestral')[a])

    sig_order = np.sort(data['Signature'].unique().astype(int)).astype(str)

    g = seaborn.FacetGrid(data, col='Signature', col_order=sig_order, col_wrap=5, margin_titles=True, sharey=False)
    g.map_dataframe(seaborn.boxplot, x='Branch', y='Proportion', fliersize=0., color='0.75')
    g.map_dataframe(seaborn.stripplot, x='Branch', y='Proportion', jitter=True, color='k',
        linewidth=0, split=False)

    for signature, ax in zip(g.col_names, g.axes):
        ax.set_title('Signature {0}\n  (p = {1:.1e})'.format(signature, test_pvalue.loc[signature]))
        yticks = ax.get_yticks()
        ax.set_yticks(yticks[yticks >= 0.])

    new_xticklabels = list()
    for label in (a.get_text() for a in g.axes[0].get_xticklabels()):
        n = len(data.loc[data['Branch'] == label, 'Sample'].unique())
        label = '{}\nn={}'.format(label, n)
        new_xticklabels.append(label)
    g.axes[0].set_xticklabels(new_xticklabels)

    seaborn.despine(offset=10, trim=True)

    plt.tight_layout()

    return g.fig


def load_signature_probabilities():
    """ Load a dataframe of cosmic signature probabilities.
    """
    sig_prob_filename = pkg_resources.resource_filename('wgs_analysis', 'data/signatures_probabilities.tsv')

    sig_prob = pd.read_csv(sig_prob_filename, sep='\t')

    sig_prob['tri_nucleotide_context'] = sig_prob['Trinucleotide']
    sig_prob['tri_nuc_idx'] = range(len(sig_prob.index))
    sig_prob['ref'] = sig_prob['Substitution Type'].apply(lambda a: a.split('>')[0])
    sig_prob['alt'] = sig_prob['Substitution Type'].apply(lambda a: a.split('>')[1])

    # Original
    sig1 = sig_prob[['tri_nuc_idx', 'ref', 'alt', 'tri_nucleotide_context']].copy()

    # Reverse complement
    sig2 = sig_prob[['tri_nuc_idx', 'ref', 'alt', 'tri_nucleotide_context']].copy()
    for col in ['ref', 'alt', 'tri_nucleotide_context']:
        sig2[col] = sig2[col].apply(reverse_complement)

    # Signatures in terms of ref and alt
    sigs = pd.concat([sig1, sig2], ignore_index=True)

    # Probability matrix
    signature_cols = filter(lambda a: a.startswith('Signature'), sig_prob.columns)
    sig_prob = sig_prob.set_index('tri_nuc_idx')[signature_cols]

    return sigs, sig_prob


def normalize_trinucleotides(data):
    """ Normalize trinucleotide reverse complementing consistent with cosmic

    Args:
        data (DataFrame): input snv data
    
    Returns:
        DataFrame: snv data with output columns added
        DataFrame: snv type, trinucleotide, ordered for plotting

    Required columns:
        - ref
        - alt
        - tri_nucleotide_context

    Output columns:
        - mutation_type
        - norm_mutation_type
        - tri_nucleotide_context
        - norm_tri_nucleotide_context
    """

    normalized_mutation_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

    norm_type_map = []

    for ref in ['A', 'C', 'T', 'G']:
        for alt in ['A', 'C', 'T', 'G']:
            if ref == alt:
                continue

            mutation_type = ref + '>' + alt
            is_revcomp = mutation_type not in normalized_mutation_types

            for left in ['A', 'C', 'T', 'G']:
                for right in ['A', 'C', 'T', 'G']:

                    trinuc = left + ref + right

                    norm_mutation_type = mutation_type
                    norm_trinuc = trinuc

                    if mutation_type not in normalized_mutation_types:
                        norm_mutation_type = complement(norm_mutation_type)
                        norm_trinuc = reverse_complement(norm_trinuc)

                    norm_type_map.append({
                        'ref': ref,
                        'alt': alt,
                        'mutation_type': mutation_type,
                        'norm_mutation_type': norm_mutation_type,
                        'tri_nucleotide_context': trinuc,
                        'norm_tri_nucleotide_context': norm_trinuc,
                    })

    norm_type_map = pd.DataFrame(norm_type_map)

    ordered_types = (
        norm_type_map[['norm_mutation_type', 'norm_tri_nucleotide_context']]
        .drop_duplicates()
        .sort_values(by=['norm_mutation_type', 'norm_tri_nucleotide_context']))

    data = data.merge(norm_type_map, on=['ref', 'alt', 'tri_nucleotide_context'], how='left')

    mismerged = data[data['norm_mutation_type'].isnull()]
    if not mismerged.empty:
        raise Exception(f'unable to normalize, example: {mismerged.iloc[0]}')

    assert not data['norm_mutation_type'].isnull().any()
    assert not data['norm_tri_nucleotide_context'].isnull().any()

    return data, ordered_types


mut_type_colors = {
    'C>A': [3/256,189/256,239/256],
    'C>G': [1/256,1/256,1/256],
    'C>T': [228/256,41/256,38/256],
    'T>A': [203/256,202/256,202/256],
    'T>C': [162/256,207/256,99/256],
    'T>G': [236/256,199/256,197/256],
}


def plot_mutation_spectra(data, count_column=None, fig=None):
    """ Plot 96 channel mutation spectra.

    Args:
        data (DataFrame): input snv data

    KwArgs:
        count_column (str): column designated already tabulated counts/proportions
    
    Returns:
        Figure: mutation spectra figure

    Required columns:
        - ref
        - alt
        - tri_nucleotide_context
    """

    data, ordered_types = normalize_trinucleotides(data)

    if count_column is None:
        plot_data = (
            data.groupby(['norm_mutation_type', 'norm_tri_nucleotide_context'])
            .size().rename('count'))
        count_column = 'count'

    else:
        plot_data = (
            data.set_index(['norm_mutation_type', 'norm_tri_nucleotide_context']))

    plot_data = plot_data.reindex(
        index=pd.MultiIndex.from_frame(ordered_types)).fillna(0).reset_index()

    plot_data['index'] = range(plot_data.shape[0])

    font = matplotlib.font_manager.FontProperties()
    font.set_family('monospace')

    if fig is None:
        fig = plt.figure(figsize=(16, 3), dpi=300)

    for mut_type, mut_type_data in plot_data.groupby('norm_mutation_type'):
        plt.bar(x='index', height=count_column, data=mut_type_data, label=mut_type, color=mut_type_colors[mut_type])
    plt.xticks(plot_data['index'], plot_data['norm_tri_nucleotide_context'], rotation=90, fontproperties=font)
    plt.xlim((-1, 97))
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    seaborn.despine(trim=True)

    return fig


def plot_cohort_mutation_signatures(
    sig_prob_filename,
    snvs_table,
    snv_nodes_table,
):
    """ Plot cohort wide clone specific mutation signatures.

    Args:
        sig_prob_filename (str): cosmic signature probability matrix
        snvs_table (pandas.DataFrame): table of per snv information including Trinucleotide
        snv_nodes_table (pandas.DataFrame): table of per snv per clone information

    """
    sigs, sig_prob = load_signature_probabilities()

    results = {}

    #
    # Per sample signatures
    #

    snvs_table = snvs_table[snvs_table['tri_nucleotide_context'].notnull()]
    snvs_table = snvs_table.merge(sigs)

    # Simple filter for variant sample presence
    snvs_table = snvs_table[snvs_table['alt_counts'] > 0]

    snvs_table['patient_sample_id'] = snvs_table['patient_id'] + '_' + snvs_table['sample_id']
    sample_sig = fit_sample_signatures(snvs_table, sig_prob, 'patient_sample_id')

    results['samples_table'] = sample_sig.copy()
    results['samples_heatmap'] = plot_signature_heatmap(sample_sig)

    #
    # Per node signatures
    #

    # Tri nucleotides from snvs table
    cohort_tri_nuc = snvs_table[['chrom', 'coord', 'ref', 'alt', 'tri_nuc_idx']].drop_duplicates()
    snv_nodes_table = snv_nodes_table.merge(cohort_tri_nuc)

    snv_nodes_table['patient_node_id'] = snv_nodes_table['patient_id'] + '_Node' + snv_nodes_table['node'].astype(str)
    node_sig = fit_sample_signatures(snv_nodes_table, sig_prob, 'patient_node_id')

    results['node_table'] = node_sig.copy()
    results['node_heatmap'] = plot_signature_heatmap(node_sig)
    results['node_signature_boxplots'] = plot_signature_boxplots(node_sig)

    return results


def calculate_tri_nucleotide_context(data, ref_genome_fasta):
    """ Calculate trinucleotide context and add as column
    """

    ref_genome = pyfaidx.Fasta(ref_genome_fasta, rebuild=False)

    data['tri_nucleotide_context'] = None

    for idx, row in data.iterrows():
        tnc = str(ref_genome[row['chrom']][row['coord']-2:row['coord']+1])
        assert len(tnc) == 3
        assert tnc[1] == row['ref']
        data.loc[idx, 'tri_nucleotide_context'] = tnc

    assert data['tri_nucleotide_context'].notnull().all()

    return data


