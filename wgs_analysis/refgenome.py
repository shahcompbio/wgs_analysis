import numpy as np
import pandas as pd
import pkg_resources


def read_chromosome_lengths(genome_fasta_index):
    fai = pd.read_csv(genome_fasta_index, sep='\t', header=None, names=['chrom', 'length', 'V3', 'V4', 'V5'])
    fai = fai.set_index('chrom')['length']
    return fai.to_dict()


class RefGenomeInfo(object):
    def __init__(self, version):
        if version == 'hg19':
            self.chromosomes = [str(a) for a in range(1, 23)] + ['X', 'Y']

            genome_fasta_index = pkg_resources.resource_filename('wgs_analysis', 'data/GRCh37-lite.fa.fai')

            self.chromosome_lengths = pd.Series(read_chromosome_lengths(genome_fasta_index)).reindex(self.chromosomes).astype(int)
            self.chromosome_lengths.index.name = 'chr'

            self.chromosome_end = np.cumsum(self.chromosome_lengths)
            self.chromosome_start = self.chromosome_end.shift(1)
            self.chromosome_start[0] = 0
            self.chromosome_start = self.chromosome_start.astype(int)
            self.chromosome_mid = (self.chromosome_start + self.chromosome_end) / 2.

            self.chromosome_info = pd.DataFrame({
                'chromosome_length': self.chromosome_lengths,
                'chromosome_end': self.chromosome_end,
                'chromosome_start': self.chromosome_start,
                'chromosome_mid': self.chromosome_mid,
            }).reset_index()

        elif version == 'grch38':
            self.chromosomes = [f'chr{a}' for a in range(1, 23)] + ['chrX', 'chrY']

            genome_fasta_index = pkg_resources.resource_filename('wgs_analysis', 'data/GRCh38.fa.fai')

            self.chromosome_lengths = pd.Series(read_chromosome_lengths(genome_fasta_index)).reindex(self.chromosomes).astype(int)
            self.chromosome_lengths.index.name = 'chr'

            self.chromosome_end = np.cumsum(self.chromosome_lengths)
            self.chromosome_start = self.chromosome_end.shift(1)
            self.chromosome_start[0] = 0
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

