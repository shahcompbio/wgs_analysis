'''
# SNV primer design

Design primers for SNVs.  Input to design is a pandas table of positions.  Sequences are created on the fly and
additional single nucleotide changes are masked if available.  Uses the design_primers module to design the primers and
return a table.

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
    return os.path.join(os.path.dirname(__file__), '{0}_snv_requirements.tsv'.format(experiment_type))


default_snv_requirements = get_requirements_filename('seqval')


def create_snv_sequence(genome_fasta, chrom, coord, variant_vcfs=[]):
    '''
    Create sequences encompassing a position, and mask variants with N
    '''
    start = max(1, coord - 200)
    end = coord + 200

    sequence = design_utils.create_sequence(genome_fasta, chrom, start, end, variant_vcfs)
    
    sequence = list(sequence)
    sequence[coord-start:coord-start+1] = ['[', 'N', ']']
    sequence = ''.join(sequence)

    return sequence


def create_snv_sequences(genome_fasta, snvs, variant_vcfs=[]):
    '''
    Generate sequences for a table of snvs
    '''
    for idx, row in snvs.iterrows():
        seq_id = row['seq_id']
        sequence = create_snv_sequence(genome_fasta, row['chrom'], row['coord'], variant_vcfs)
        yield seq_id, sequence


def snv_primer_product_check(snvs, primers, products):
    '''
    Ensure the product overlaps the SNV, add product start and end to the primer table
    '''
    snv_products = products[['seq_id', 'primer_id', 'chrom', 'start', 'end']].merge(snvs[['seq_id', 'chrom', 'coord']].drop_duplicates(), on=['seq_id', 'chrom'], how='inner')
    snv_products = snv_products[(snv_products['coord'] > snv_products['start']) & (snv_products['coord'] < snv_products['end'])]
    snv_products = snv_products.rename(columns={'start':'product_start', 'end':'product_end'})
    primers = primers.merge(snv_products[['seq_id', 'primer_id', 'product_start', 'product_end']].drop_duplicates(), on=['seq_id', 'primer_id'], how='inner')
    products = products.merge(snv_products[['seq_id', 'primer_id']].drop_duplicates(), how='inner')
    return primers, products


def design(genome_filename, snvs, variant_vcfs=[], requirements_filename=default_snv_requirements, max_stage=-1, max_primers=-1, primer3_params=None):
    """
    Design primers for an SNV validation experiment

    Args:
        genome_filename(str) : genome file in fasta format
        snvs(pandas.DataFrame) : table of SNV information

    KwArgs:
        variant_vcfs(list) : list of VCF files with germline variants
        requirements_filename(str) : requirements tsv file detailing design requirements by stage
        max_stage(int) : maximum stage before failing, -1 to try all stages
        max_primers(int) : maximum number of primers to design before returning
        primer3_params(dict) : Additional Primer 3 Parameters

    Returns:
        pandas.DataFrame of primer information

    The table of SNVs should have the following required columns:
     * seq_id : unique identifier for the SNV
     * chrom : chromosome of the SNV
     * coord : coordinate of the SNV

    """
    snv_sequences = create_snv_sequences(genome_filename, snvs, variant_vcfs)
    design_snv_callback = functools.partial(snv_primer_product_check, snvs)

    primer_table = design_primers.design(requirements_filename, snv_sequences, design_callback=design_snv_callback, max_stage=max_stage, max_primers=max_primers, primer3_params=primer3_params)

    primer_table = primer_table.merge(snvs, on='seq_id', how='inner')

    return primer_table



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('genome', help='genome fasta filename')
    argparser.add_argument('snvs', help='snvs table filename')
    argparser.add_argument('primers', help='primer table filename')
    argparser.add_argument('--requirements', required=False, default=default_snv_requirements, help='design requirements table')
    argparser.add_argument('--varvcfs', nargs='+', required=False, default=[], help='variant vcfs for masking')
    args = argparser.parse_args()

    snvs = pd.read_csv(args.snvs, sep='\t', converters={'chrom':str})[['chrom', 'coord']].drop_duplicates()

    with blat_server.BlatServer([args.genome]):
        primer_table = design(args.genome, snvs, args.varvcfs)

    primer_table.to_csv(args.primers, sep='\t', index=False, na_rep='NA')





