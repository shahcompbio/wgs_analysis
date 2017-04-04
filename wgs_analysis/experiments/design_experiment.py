import os
import warnings
import random
import pandas as pd

import wgs_analysis.experiments.design_primers as design_primers
import wgs_analysis.experiments.design_snv_primers as design_snv_primers
import wgs_analysis.experiments.design_break_primers as design_break_primers
import blat_server



def design_categories(selection, designer, genome_fasta, seed=2014, primer3_params=None, **kwargs):
    """ Design primers in groups with group specific requirements

    Args:
        selection(pandas.DataFrame) : selected variants table, see format below
        designer(callable) : design function, either design_snv_primers.design or design_break_primers.design
        genome_fasta(str) : genome fasta filename

    KwArgs:
        seed(int) : random seed for shuffling
        primer3_params(dict) : Additional Primer 3 Parameters
        kwargs : additional arguments to design function

    Returns:
        pandas.DataFrame of primers

    The selection table requires the following columns:
     * seq_id : unique identifier for the SNV/breakpoint
     * category : reason the SNV/breakpoint was selected
     * max_primer_stage : maximum stage at which primer selection will fail, -1 to try all stages
     * subset_count : number of SNVs/breakpoints to design from this category

    """

    random.seed(2014)

    primer_table = list()

    excluded_events = set()

    # Design SNV primers for each category
    for category, variants in selection.groupby('category'):

        # Remove variants for which we have already designed primers
        variants = variants[~variants['seq_id'].isin(excluded_events)]

        # Check if any variants remain to be designed for
        if len(variants.index) == 0:
            continue

        subset_count = variants['subset_count'].iloc[0]
        max_primer_stage = variants['max_primer_stage'].iloc[0]
        
        # Shuffle variants within this category
        shuffled_index = list(variants.index.values)
        random.shuffle(shuffled_index)
        variants = variants.ix[shuffled_index]

        # Design primers
        primers = designer(genome_fasta, variants, max_stage=max_primer_stage, max_primers=subset_count, primer3_params=primer3_params, **kwargs)

        # Warn if requirements not met
        if subset_count >= 0:
            if len(primers.index) < subset_count:
                warnings.warn('insufficient variants for category ' + category)
        else:
            for seq_id in variants['seq_id'].values:
                if seq_id not in primers['seq_id'].values:
                    warnings.warn('no primers for variant ' + str(seq_id) + ' in category ' + category)

        primer_table.append(primers)

        excluded_events.update(primers['seq_id'].values)

    primer_table = pd.concat(primer_table, ignore_index=True).drop_duplicates()

    for seq_id, primers in primer_table.groupby('seq_id'):
        if len(primers) > 1:
            warnings.warn('designed ' + str(len(primers)) + ' primers for sequence ' + str(seq_id) + ' in categories ' + ', '.join(primers['category']))

    return primer_table



def print_design_stats(**kwargs):
    """
    Print the number of events for which primers will be designed

    Args:
        kwargs : pandas.DataFrame tables of events with the format as described for design_seqval

    """

    primer_pair_count = 0

    for event_type, events in kwargs.iteritems():

        if events is None:
            continue

        for category, category_events in events.groupby('category'):

            subset_count = category_events['subset_count'].iloc[0]
            if subset_count < 0:
                subset_count = len(category_events.index)

            total_count = len(category_events.index)

            primer_pair_count += subset_count

            print 'designing {0}/{1} for {2} {3}'.format(subset_count, total_count, category, event_type)

    print 'designing {0} primer pairs'.format(primer_pair_count)



def print_experiment_stats(**kwargs):
    """
    Print the number of events for which primers have been designed

    Args:
        kwargs : pandas.DataFrame tables of events with the format as described for design_seqval

    """

    primer_pair_count = 0

    for event_type, events in kwargs.iteritems():

        if events is None:
            continue

        for category, category_events in events.groupby('category'):
            total_count = len(category_events.index)

            primer_pair_count += total_count

            print 'designed {0} for {1} {2}'.format(total_count, category, event_type)

    print 'designing {0} primer pairs'.format(primer_pair_count)



def design_experiment(patient_id, experiment_id, experiment_type, genome_fasta, variant_vcfs, snvs=None, breakpoints=None, primer3_params=None):
    """
    Design primers for both SNVs and breakpoints and write to the appropriate location in the
    meta_data directory.

    Args:
        patient_id(str) : patient identifier
        experiment_id(str) : experiment identifier to distinguish different experiments per patient
        experiment_type(str) : type of experiment, either `seqval` or `singlecell`
        genome_fasta(str) : genome fasta filename
        variant_vcfs(list) : list of vcfs containing variants to avoid

    KwArgs:
        snvs(pandas.DataFrame) : SNVs selected for primer design
        breakpoints(pandas.DataFrame) : Breakpoints selected for primer design
        primer3_params(dict) : Additional Primer 3 Parameters

    Breakpoints and SNVs are given as pandas tables with the following required columns:
     * seq_id : unique identifier for the SNV/breakpoint
     * category : reason the SNV/breakpoint was selected
     * max_primer_stage : maximum stage at which primer selection will fail, -1 to try all stages
     * subset_count : number of SNVs/breakpoints to design from this category

    Additionally, SNVs require the following columns:
     * chrom : chromosome of SNV
     * coord : coordinate of the SNV
     * ref : reference base of the SNV
     * alt : variant base of the SNV

    Additionally, breakpoint tables are 2 rows per breakpoint, one for each breakend, and require the
    following columns:
     * chrom : pair of chromosomes, comma separated
     * strand : pair of strands, comma separated
     * coord : pair of coordinates, comma separated

    """

    # Check args

    if experiment_type not in ('seqval', 'singlecell'):
        raise ValueError('unsupported experiment type')

    required_snv_columns = set(['seq_id', 'category', 'max_primer_stage', 'subset_count',
                                'chrom', 'coord', 'ref', 'alt'])

    required_breakpoint_columns = set(['seq_id', 'category', 'max_primer_stage', 'subset_count',
                                       'chrom', 'strand', 'coord'])

    if snvs is not None and not required_snv_columns.issubset(snvs.columns.values):
        raise ValueError('missing columns from snv table')

    if breakpoints is not None and not required_breakpoint_columns.issubset(breakpoints.columns.values):
        raise ValueError('missing columns from breakpoint table')


    # Design primers

    print_design_stats(snvs=snvs, breakpoints=breakpoints)

    primer_table = list()

    with blat_server.BlatServer(genome_fasta):

        if snvs is not None:

            print 'designing primers for SNVs'

            snv_primers = design_categories(snvs, design_snv_primers.design,
                genome_fasta, variant_vcfs=variant_vcfs,
                requirements_filename=design_snv_primers.get_requirements_filename(experiment_type),
                primer3_params=primer3_params)

            snv_primers['variant_type'] = 'snv'
            snv_primers['strand'] = '+'

            primer_table.append(snv_primers)

        if breakpoints is not None:

            print 'designing primers for breakpoints'

            breakpoint_primers = design_categories(breakpoints, design_break_primers.design,
                genome_fasta, variant_vcfs=variant_vcfs,
                requirements_filename=design_break_primers.get_requirements_filename(experiment_type),
                primer3_params=primer3_params)

            breakpoint_primers['variant_type'] = 'breakpoint'
            breakpoint_primers['ref'] = 'NA'
            breakpoint_primers['alt'] = 'NA'

            primer_table.append(breakpoint_primers)


    primer_table = pd.concat(primer_table, ignore_index=True)


    # Merge duplicates, created comma delimted list in category column

    def merge_duplicates_pairs(df):
        if len(df.index) == 1:
            return df
        else:
            categories = ','.join(df['category'])
            df = df.sort('stage')
            df = df.iloc[0:1]
            df['category'] = categories
            return df

    primer_table = primer_table.groupby('seq_id').apply(merge_duplicates_pairs)

    primer_table['patient_id'] = patient_id

    print 'designed {} primers'.format(len(primer_table.index))

    print_experiment_stats(
        snvs=primer_table[primer_table['variant_type'] == 'snv'],
        breakpoints=primer_table[primer_table['variant_type'] == 'breakpoint'])

    return primer_table



def create_snv_seq_id(snvs):
    return snvs.apply(lambda row: '{0}:{1}'.format(row['chrom'], row['coord']), axis=1)



def create_snvs_table(selection, snvs):
    """
    Create a breakpoint table compatible with design_seqval

    Args:
        selection(pandas.DataFrame) : selected snvs table, chrom and coord columns required
        snvs(pandas.DataFrame) : full SNVs table

    Returns:
        pandas.DataFrame selection breakpoints with all columns from selection and the following additional columns:
         * seq_id
         * ref
         * alt

    """

    selection_columns = list(selection.columns.values)
    additional_columns = ['ref', 'alt']

    snvs = snvs.merge(selection, on=['chrom', 'coord'], how='inner')
    snvs = snvs[selection_columns + additional_columns].drop_duplicates()
    snvs['seq_id'] = create_snv_seq_id(snvs)

    return snvs



def create_breakpoint_seq_id(snvs):
    return snvs.apply(lambda row: row['prediction_id'], axis=1)



def create_breakpoints_table(selection, breakpoints):
    """
    Create a breakpoint table compatible with design_seqval

    Args:
        selection(pandas.DataFrame) : selected breakpoints table, prediction_id column required
        breakpoints(pandas.DataFrame) : full breakpoints table

    Returns:
        pandas.DataFrame selection breakpoints with all columns from selection and the following additional columns:
         * seq_id
         * chrom : pair of chromosomes, comma separated
         * strand : pair of strands, comma separated
         * coord : pair of coordinates, comma separated
  
    """

    breakpoints = breakpoints.merge(selection[['prediction_id']].drop_duplicates(), how='inner').set_index('prediction_id')

    def calculate_chrom(row):
        return '{0},{1}'.format(row['chromosome_1'], row['chromosome_2'])
        
    def calculate_strand(row):
        return '{0},{1}'.format(row['strand_1'], row['strand_2'])
        
    def calculate_coord(row):
        return '{0},{1}'.format(row['position_1'], row['position_2'])

    chrom = breakpoints.apply(calculate_chrom, axis=1)
    strand = breakpoints.apply(calculate_strand, axis=1)
    coord = breakpoints.apply(calculate_coord, axis=1)

    breakpoints = pd.DataFrame({'chrom':chrom, 'strand':strand, 'coord':coord}).reset_index()

    breakpoints = breakpoints.merge(selection, on='prediction_id', how='inner')

    breakpoints['seq_id'] = create_breakpoint_seq_id(breakpoints)
    breakpoints = breakpoints.drop('prediction_id', axis=1)

    return breakpoints



def calculate_seq_id_category_map(primer_table):
    """
    Given a primer table calculate a mapping between seq_id and categories

    Args:
        primer_table(pandas.DataFrame) : data frame with seq_id and category as mandatory columns

    Returns:
        pandas.DataFrame with seq_id and category columns

    We are storing categories as comma separated lists in the primer table.  This helper function extracts
    a one to many mapping between seq_ids and individual categories. 

    """

    seq_id_category = list()
    for seq_id, categories in primer_table[['seq_id', 'category']].values:
        for category in categories.split(','):
            seq_id_category.append((seq_id, category))
            
    return pd.DataFrame(seq_id_category, columns=['seq_id', 'category'])


