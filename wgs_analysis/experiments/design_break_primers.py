'''
# Breakpoint primer design

Design primers for breakpoint sequences.  Input to design is a pandas table of breakpoints.  Breakpoint sequences are
created on the fly and single nucleotide changes are masked if available.  Uses the design_primers module to design the
primers and return a table.

'''

import os
import functools
import argparse
import subprocess
import pandas as pd

import design_utils
import design_primers
import blat_server


def get_requirements_filename(experiment_type):
    return os.path.join(os.path.dirname(__file__), '{0}_breakpoint_requirements.tsv'.format(experiment_type))


default_breakpoint_requirements = get_requirements_filename('seqval')


def create_breakpoint_sequences(genome_fasta, breakpoints, variant_vcfs=[]):
    '''
    Generate sequences for a table of breakpoints
    '''
    for idx, row in breakpoints.iterrows():
        seq_id = row['seq_id']
        chroms = row['chrom'].split(',')
        strands = row['strand'].split(',')
        coords = [int(a) for a in row['coord'].split(',')]
        sequences = []
        for side, target_strand in zip((0, 1), ('+', '-')):
            if strands[side] == '+':
                start = coords[side] - 300 + 1
                end = coords[side]
            else:
                start = coords[side]
                end = coords[side] + 300 - 1
            sequence = design_utils.create_sequence(genome_fasta, chroms[side], start, end, variant_vcfs)
            if strands[side] != target_strand:
                sequence = design_utils.reverse_complement(sequence)
            sequences.append(sequence)
        yield seq_id, sequences[0] + '[]' + sequences[1]


def design(genome_filename, breakpoints, variant_vcfs=[], requirements_filename=default_breakpoint_requirements, max_stage=-1, max_primers=-1, primer3_params=None):
    """
    Design primers for an breakpoint validation experiment

    Args:
        genome_filename(str) : genome file in fasta format
        breakpoints(pandas.DataFrame) : table of breakpoint information

    KwArgs:
        variant_vcfs(list) : list of VCF files with germline variants
        requirements_filename(str) : requirements tsv file detailing design requirements by stage
        max_stage(int) : maximum stage before failing, -1 to try all stages
        max_primers(int) : maximum number of primers to design before returning
        primer3_params(dict) : Additional Primer 3 Parameters

    Returns:
        pandas.DataFrame of primer information

    The table of breakpoints should have the following required columns:
     * seq_id : unique identifier for the breakpoint
     * chrom : pair of chromosomes, comma separated
     * strand : pair of strands, comma separated
     * coord : pair of coordinates, comma separated

    """
    breakpoint_sequences = create_breakpoint_sequences(genome_filename, breakpoints, variant_vcfs)

    primer_table = design_primers.design(requirements_filename, breakpoint_sequences, max_stage=max_stage, max_primers=max_primers, primer3_params=primer3_params)

    if len(primer_table.index) == 0:
        return pd.DataFrame(columns=['seq_id']).astype(int)

    def calculate_product_start(row):
        return row['sequence'].index(row['left_primer']) + 1

    def calculate_product_end(row):
        return row['sequence'].index(design_utils.reverse_complement(row['right_primer'])) + len(row['right_primer'])

    def recalculate_sequence(row):
        return row['sequence'][row['product_start'] - 1 : row['product_end']]

    def check_recalculated(row):
        return row['product_sequence'].startswith(row['left_primer']) and \
               row['product_sequence'].endswith(design_utils.reverse_complement(row['right_primer'])) and \
               len(row['product_sequence']) == int(row['product_size'])

    primer_table['product_start'] = primer_table.apply(calculate_product_start, axis=1)
    primer_table['product_end'] = primer_table.apply(calculate_product_end, axis=1)
    primer_table['product_sequence'] = primer_table.apply(recalculate_sequence, axis=1)
    primer_table['product_check'] = primer_table.apply(check_recalculated, axis=1)

    assert primer_table['product_check'].all()

    primer_table = primer_table.drop(['product_sequence', 'product_check'], axis=1)

    primer_table = primer_table.merge(breakpoints, on='seq_id', how='inner')

    return primer_table



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('genome', help='genome fasta or 2bit filename')
    argparser.add_argument('breakpoints', help='breakpoints table filename')
    argparser.add_argument('primers', help='primer table filename')
    argparser.add_argument('--requirements', required=False, default=default_breakpoint_requirements, help='design requirements table')
    argparser.add_argument('--varvcfs', nargs='+', required=False, default=[], help='variant vcfs for masking')
    args = argparser.parse_args()

    breakpoints = pd.read_csv(args.breakpoints, sep='\t', converters={'chrom':str})[['chrom', 'coord']].drop_duplicates()

    with blat_server.BlatServer(args.genome):
        primer_table = design(args.genome, breakpoints, args.varvcfs)

    primer_table.to_csv(args.primers, sep='\t', index=False, na_rep='NA')





