'''
# Design primers utils

Provides convencience functions used by primer design modules.

'''


import os
import shutil
import subprocess
import string
import uuid


def get_region_string(chrom, start, end):
    return '{0}:{1}-{2}'.format(chrom, int(start), int(end))

def fetch_sequence(genome, chrom, start, end):
    faidx_result = str(subprocess.check_output(['samtools', 'faidx', genome, get_region_string(chrom, start, end)]))
    seq_id, sequence = next(read_fasta(faidx_result.splitlines()))
    return sequence

def fetch_variants(vcf_filename, chrom, start, end):
    for line in subprocess.check_output(['tabix', vcf_filename, get_region_string(chrom, start, end)]).splitlines():
        if line == '':
            continue
        row = line.split()
        assert chrom == row[0]
        coord = int(row[1])
        ref = row[3]
        alt = row[4]
        length = len(ref)
        variant_start = max(start, coord)
        variant_end = min(end, coord + length - 1)
        yield variant_start, variant_end - variant_start + 1

def reverse_complement(sequence):
    return sequence[::-1].translate(string.maketrans('ACTGactg','TGACtgac'))

def read_fasta(fasta):
    ''' Read sequences from an open fasta file. Yield name, sequence pairs '''
    name = None
    sequences = []
    for line in fasta:
        line = line.rstrip()
        if len(line) == 0:
            continue
        if line[0] == '>':
            if name is not None:
                yield (name, ''.join(sequences))
            name = line[1:]
            sequences = []
        else:
            sequences.append(line)
    if name is not None:
        yield (name, ''.join(sequences))

def create_sequence(genome_fasta, chrom, start, end, variant_vcfs=[]):
    '''
    Create sequences spanning a breakpoint, and mask variants with N
    '''
    sequence = list(fetch_sequence(genome_fasta, chrom, start, end))

    for var_vcf in variant_vcfs:
        for var_start, var_length in fetch_variants(var_vcf, chrom, start, end):
            local_var_start = max(0, var_start - start)
            sequence[local_var_start:local_var_start+var_length] = ['N'] * var_length
    
    sequence = ''.join(sequence)

    return sequence

class TempDirectory(object):
    """ Temporary directory context manager """
    def __init__(self):
        self.name = os.path.abspath(str(uuid.uuid4()).replace('-', ''))
    def __enter__(self):
        os.mkdir(self.name)
        return self.name
    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.name)

