"""
SNV annotation functionality

Annotations for SNV data.

Functions take pandas tables and add additional columns.
"""

import itertools
import pandas as pd

def annotate_cluster(snvs, bandwidth=5000, min_spacing=4):
    """ Add cluster_id and cluster_size columnns for clusters of snvs
    within bandwidth nt of each other.  cluster_size counts only those snvs
    that are at least min_spacing apart
    """
    class PositionClusterKey(object):
        def __init__(self, bandwidth):
            self.bandwidth = bandwidth
            self.start = None
            self.previous = None
        def __call__(self, position):
            if self.start is None or position - self.previous > self.bandwidth:
                self.start = position
            self.previous = position
            return self.start
    cluster_id = 0
    clusters_table = list()
    for chromosome, chr_group in snvs[['chrom', 'coord']].drop_duplicates().groupby('chrom'):
        for start, pos_group in itertools.groupby(sorted(chr_group['coord']), key=PositionClusterKey(bandwidth)):
            pos_group = list(pos_group)
            cluster_size = 0
            previous_position = None
            for position in sorted(pos_group):
                if previous_position is None or position - previous_position >= min_spacing:
                    cluster_size += 1
                previous_position = position
            for position in pos_group:
                clusters_table.append((chromosome, position, cluster_id, cluster_size))
            cluster_id += 1
    clusters_table = pd.DataFrame(clusters_table, columns=['chrom', 'coord', 'cluster_id', 'cluster_size'])
    snvs = snvs.merge(clusters_table, left_on=['chrom', 'coord'], right_on=['chrom', 'coord'])
    return snvs


def annotate_kataegis(snvs, min_cluster_size=5):
    """ Annotate kataegis_id column, identifier for the kataegis event the snv
    is putatively part of
    """

    snvs = annotate_cluster(snvs)

    kataegis_id = 0
    snvs['kataegis_id'] = None
    for cluster_id, group in snvs.groupby('cluster_id'):
        if group['cluster_size'].iloc[0] < min_cluster_size:
            continue
        for idx, row in group.iterrows():
            snvs['kataegis_id'][idx] = kataegis_id
        kataegis_id += 1
    return snvs


def shortened_site_names(row):
    shortened = list()
    for site_id in sorted(row.loc[row['alt_counts'] > 0, 'site_id'].unique()):
        shortened.append(''.join(a[0] for a in site_id.split('_')))
    return '_'.join(shortened)


def annotate_site_names(snvs):
    """ Annotate unique identifier to each presence/absence group
    """
    snvs.set_index(['chrom', 'coord'], inplace=True)
    snvs['site_names'] = snvs.groupby(level=[0, 1]).apply(shortened_site_names)
    snvs.reset_index(inplace=True)

    return snvs

