'''
# Design primers module

Provides functionality for designing primers based on input sequences provided in a pandas table.  A blat server must be
running, and the convencience class BlatServer provides a context manager based server for a given genome.  Runs primer3
and isPCR to obtain primers, and applies constraints to filter primer pairs.  Progressively relaxes constraints if
requested.

'''

import argparse
import csv
import collections
import sys
import os
import tempfile
import time
import math
import subprocess
import numpy as np
import pandas as pd
import distutils.spawn

import design_utils
import blat_server


primer3_core_ex = distutils.spawn.find_executable('primer3_core')
if primer3_core_ex is None:
    raise Exception('primer3_core not found')

primer3_dir = os.path.dirname(os.path.realpath(primer3_core_ex))

default_primer3_parameters = {
    'PRIMER_MIN_SIZE': 18,
    'PRIMER_MAX_SIZE': 26,
    'PRIMER_NUM_NS_ACCEPTED': 0,
    'PRIMER_FILE_FLAG': 0,
    'PRIMER_PICK_INTERNAL_OLIGO': 0,
    'PRIMER_MIN_TM': 58.0,
    'PRIMER_OPT_TM': 60.0,
    'PRIMER_MAX_TM': 62.0,
    'PRIMER_MAX_POLY_X': 4,
    'PRIMER_THERMODYNAMIC_TEMPLATE_ALIGNMENT': 1,
    'PRIMER_THERMODYNAMIC_PARAMETERS_PATH': '{primer3_dir}/primer3_config/'.format(primer3_dir=primer3_dir),
    'PRIMER_MISPRIMING_LIBRARY': '{primer3_dir}/humrep_and_simple.txt'.format(primer3_dir=primer3_dir),
}


def run_primer3(sequence, req, params=None):
    '''
    Run primer 3 and obtain a list of potential primers.
    '''
    target_start = sequence.index('[')
    target_size = sequence.index(']') - sequence.index('[') - 1
    sequence = sequence.replace('[', '').replace(']', '')

    req_target_start = target_start - int(req['target_buffer'])
    req_target_size = target_size + 2 * int(req['target_buffer'])

    primers = list()

    primer3_parameters = default_primer3_parameters.copy()
    primer3_parameters['PRIMER_PRODUCT_SIZE_RANGE'] = '{0}-{1}'.format(req['min_product_size'], req['max_product_size'])
    primer3_parameters['PRIMER_OPT_SIZE'] = '{0}'.format(req['opt_size'])
    if req['gc_clamp'] >= 0:
        primer3_parameters['PRIMER_GC_CLAMP'] = '{0}'.format(req['gc_clamp'])

    if params is not None:
        primer3_parameters.update(params)

    primer3_proc = subprocess.Popen(['primer3_core'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    primer3_proc.stdin.write('SEQUENCE_TEMPLATE={0}\n'.format(sequence))
    primer3_proc.stdin.write('SEQUENCE_TARGET={0},{1}\n'.format(req_target_start, req_target_size))
    for key, value in primer3_parameters.iteritems():
        primer3_proc.stdin.write('{}={}\n'.format(key, value))
    primer3_proc.stdin.write('=\n')
    primer3_proc.stdin.close()

    pr3 = dict(line.rstrip().split('=') for line in primer3_proc.stdout)
    primer3_proc.stdout.close()

    if primer3_proc.wait() != 0:
        sys.stderr.write('primer3 errors\n')
        sys.exit(1)

    for primer_id in range(0, 5):
        if 'PRIMER_LEFT_{0}_SEQUENCE'.format(primer_id) in pr3:
            primer_info = collections.OrderedDict()
            primer_info['primer_id'] = primer_id
            primer_info['sequence'] = sequence
            primer_info['left_primer'] = pr3['PRIMER_LEFT_{0}_SEQUENCE'.format(primer_id)]
            primer_info['right_primer'] = pr3['PRIMER_RIGHT_{0}_SEQUENCE'.format(primer_id)]
            primer_info['left_primer_length'] = pr3['PRIMER_LEFT_{0}'.format(primer_id)].split(',')[1]
            primer_info['right_primer_length'] = pr3['PRIMER_RIGHT_{0}'.format(primer_id)].split(',')[1]
            primer_info['left_tm'] = pr3['PRIMER_LEFT_{0}_TM'.format(primer_id)]
            primer_info['right_tm'] = pr3['PRIMER_RIGHT_{0}_TM'.format(primer_id)]
            primer_info['product_size'] = pr3['PRIMER_PAIR_{0}_PRODUCT_SIZE'.format(primer_id)]
            primer_info['target_buffer'] = req['target_buffer']
            primer_info['gc_clamp'] = req['gc_clamp']
            primer_info['stage'] = req['stage']
            primers.append(primer_info)

    primers = pd.DataFrame(primers)

    return primers


def run_insilico_pcr(primers):
    '''
    Use gfPcr to count the number of potential products in the human genome.
    Add the product count to the list of dictionaries in primers.
    '''
    products = list()

    with design_utils.TempDirectory() as temps_dir:

        temp_input_filename = os.path.join(temps_dir, 'primers.tsv')
        temp_output_filename = os.path.join(temps_dir, 'products.fa')

        with open(temp_input_filename, 'w') as temp_input_file:
            for idx, primer in primers.iterrows():
                temp_input_file.write('{0}\t{1}\t{2}\t4000\n'.format(primer['primer_id'], primer['left_primer'], primer['right_primer']))

        is_pcr_proc = subprocess.check_call(['gfPcr', 'localhost', '8899', '/', temp_input_filename, temp_output_filename, '-out=psl'], stdout=subprocess.PIPE)

        with open(temp_output_filename, 'r') as temp_output_file:
            for line in temp_output_file:
                row = line.split()
                primer_id = int(row[9])
                chrom = row[13]
                start = int(row[15]) + 1
                end = int(row[16])
                products.append({'primer_id':primer_id, 'chrom':chrom, 'start':start, 'end':end})

    products = pd.DataFrame(products)

    if len(products.index) == 0:
        products = pd.DataFrame(columns=['primer_id', 'chrom', 'start', 'end'])

    return products


def run_alignment(primers):
    '''
    Use gfServer to count the number of alignments of each primer in the human genome.
    Add the alignment count to the list of dictionaries in primers.
    '''
    alignment_infos = list()

    with design_utils.TempDirectory() as temps_dir:

        temp_fasta_filename = os.path.join(temps_dir, 'primers.fa')
        temp_psl_filename = os.path.join(temps_dir, 'alignments.psl')

        with open(temp_fasta_filename, 'w') as temp_fasta_file:
            for idx, primer in primers.iterrows():
                temp_fasta_file.write('>{0}_left\n{1}\n'.format(idx, primer['left_primer']))
                temp_fasta_file.write('>{0}_right\n{1}\n'.format(idx, primer['right_primer']))

        subprocess.check_call(['gfClient', 'localhost', '8899', '/', temp_fasta_filename, temp_psl_filename, '-minScore=0', '-minIdentity=0', '-nohead'], stdout=subprocess.PIPE)

        with open(temp_psl_filename, 'r') as temp_psl_file:
            for row in csv.reader(temp_psl_file, delimiter='\t'):
                matches = row[0]
                length = row[10]
                if matches != length:
                    continue
                idx, side = row[9].split('_')
                idx = int(idx)
                primer_id = primers.loc[idx, 'primer_id']
                alignment_infos.append({'primer_id':primer_id, side+'_alignment_count':1})

    alignment_infos = pd.DataFrame(alignment_infos)

    alignment_infos = alignment_infos.reindex(columns=['primer_id', 'left_alignment_count', 'right_alignment_count'])

    alignment_infos = alignment_infos.fillna(0)
    alignment_infos['left_alignment_count'] = alignment_infos['left_alignment_count'].astype(int)
    alignment_infos['right_alignment_count'] = alignment_infos['right_alignment_count'].astype(int)

    alignment_infos = alignment_infos.groupby('primer_id').sum().reset_index()

    return alignment_infos


def pick_primers(requirements, seq_data, design_callback=None, max_stage=-1, max_primers=-1, primer3_params=None):
    '''
    Design primers, iteratively relaxing design requirements
    '''
    primer_table = list()

    for idx, (seq_id, sequence) in enumerate(seq_data):

        for req_idx, req_row in requirements.iterrows():

            if max_stage >= 0 and req_row['stage'] > max_stage:
                continue

            primers = run_primer3(sequence, req_row, params=primer3_params)
            primers['seq_id'] = seq_id

            if len(primers.index) == 0:
                continue

            def check_primer_uniqueness(row):
                if row['sequence'].count(row['left_primer']) != 1:
                    return False
                if row['sequence'].count(design_utils.reverse_complement(row['right_primer'])) != 1:
                    return False
                return True

            primers = primers[primers.apply(check_primer_uniqueness, axis=1)]

            if len(primers.index) == 0:
                continue

            products = run_insilico_pcr(primers)
            products['seq_id'] = seq_id

            primers.set_index('primer_id', inplace=True)
            primers['product_count'] = products.groupby('primer_id').size()
            primers['product_count'] = primers['product_count'].fillna(0).astype(int)
            primers.reset_index(inplace=True)

            alignment_info = run_alignment(primers)
            primers = primers.merge(alignment_info, on='primer_id').fillna(0)

            if req_row['min_product_count'] >= 0:
                primers = primers[primers['product_count'] >= req_row['min_product_count']]

            if req_row['max_product_count'] >= 0:
                primers = primers[primers['product_count'] <= req_row['max_product_count']]

            if req_row['max_alignment_count'] >= 0:
                primers = primers[primers['left_alignment_count'] <= req_row['max_alignment_count']]
                primers = primers[primers['right_alignment_count'] <= req_row['max_alignment_count']]

            if design_callback is not None:
                primers, products = design_callback(primers, products)

            if len(primers.index) > 0:
                primer_table.append(primers.iloc[0:1])
                sys.stderr.write('+')
                break
            else:
                sys.stderr.write('.')

        if max_primers >= 0 and len(primer_table) >= max_primers:
            break

    sys.stderr.write('\n')

    if len(primer_table) == 0:
        return pd.DataFrame(columns=[
            'primer_id', 'gc_clamp', 'left_primer', 'left_primer_length',
            'left_tm', 'product_size', 'right_primer', 'right_primer_length',
            'right_tm', 'sequence', 'stage', 'target_buffer', 'seq_id',
            'product_count', 'left_alignment_count', 'right_alignment_count',
            'product_start', 'product_end'])

    primer_table = pd.concat(primer_table, ignore_index=True)

    return primer_table


def design(requirements_filename, sequences, design_callback=None, max_stage=-1, max_primers=-1, primer3_params=None):

    requirements = pd.read_csv(requirements_filename, sep='\t')
    primer_table = pick_primers(requirements, sequences, design_callback=design_callback, max_stage=max_stage, max_primers=max_primers, primer3_params=primer3_params)
    return primer_table


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('genome', help='genome fasta filename')
    argparser.add_argument('requirements', help='design requirements table')
    argparser.add_argument('sequences', help='sequences fasta filename')
    argparser.add_argument('primers', help='primer table filename')
    args = argparser.parse_args()

    with blat_server.BlatServer(args.genome), open(args.sequences, 'r') as sequences_file:
        primer_table = design(args.requirements, design_utils.read_fasta(sequences_file))
        primer_table.to_csv(args.primers, sep='\t', index=False, na_rep='NA')
