import numpy as np
import pandas as pd
from importlib.resources import files


def read_chromosome_lengths(genome_fasta_index):
    fai = pd.read_csv(genome_fasta_index, sep='\t', header=None, names=['chrom', 'length', 'V3', 'V4', 'V5'])
    fai = fai.set_index('chrom')['length']
    return fai.to_dict()


# Chromosomes in plotting order, and the shipped fasta index they are measured
# from, per supported genome version
genome_versions = {
    'hg19': {
        'chromosomes': [str(a) for a in range(1, 23)] + ['X', 'Y'],
        'genome_fasta_index': 'GRCh37-lite.fa.fai',
    },
    'grch38': {
        'chromosomes': [f'chr{a}' for a in range(1, 23)] + ['chrX', 'chrY'],
        'genome_fasta_index': 'GRCh38.fa.fai',
    },
    'mm10': {
        'chromosomes': [str(a) for a in range(1, 20)] + ['X', 'Y'],
        'genome_fasta_index': 'mm10.fa.fai',
    },
}


class RefGenomeInfo(object):
    def __init__(self, version):
        if version not in genome_versions:
            raise ValueError(f'unknown genome version {version}, expected one of {sorted(genome_versions)}')

        self.version = version
        self.chromosomes = list(genome_versions[version]['chromosomes'])

        genome_fasta_index = str(files('wgs_analysis').joinpath(
            'data', genome_versions[version]['genome_fasta_index']))

        self.chromosome_lengths = pd.Series(read_chromosome_lengths(genome_fasta_index)).reindex(self.chromosomes).astype(int)
        self.chromosome_lengths.index.name = 'chr'

        self.chromosome_end = np.cumsum(self.chromosome_lengths)
        self.chromosome_start = self.chromosome_end.shift(1)
        self.chromosome_start.iloc[0] = 0
        self.chromosome_start = self.chromosome_start.astype(int)
        self.chromosome_mid = (self.chromosome_start + self.chromosome_end) / 2.

        self.chromosome_info = pd.DataFrame({
            'chromosome_length': self.chromosome_lengths,
            'chromosome_end': self.chromosome_end,
            'chromosome_start': self.chromosome_start,
            'chromosome_mid': self.chromosome_mid,
        }).reset_index()


info = None


def set_genome_version(version):
    global info
    info = RefGenomeInfo(version)

set_genome_version('hg19')
