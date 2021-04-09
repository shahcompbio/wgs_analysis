
import sys
import csv
import subprocess
import math
import argparse
import os
import itertools
import string
import gzip
import pandas as pd
import numpy as np
from scipy import stats


###
# Annnotation and filtering parameters
###

align_prob_score_min = 0.9
valid_prob_score_min = 0.5
chimeric_prob_score_min = 0.9

high_conf_min_tumour_count_max = 4
high_conf_min_num_split = 2

max_other_libraries = 0



###
# Helper functions
###

def get_counts_columns(breakpoints):

    return filter(lambda a: a.endswith('_count'), breakpoints.columns)


def shortened_site_names(row):
    shortened = list()
    for site_id in sorted(row.loc[row['num_reads'] > 0, 'site_id'].unique()):
        shortened.append(''.join(a[0] for a in site_id.split('_')))
    return '_'.join(shortened)


def calculate_site_names(breakpoint_site):
    """ Calculate unique identifier to each presence/absence group
    """
    return breakpoint_site.set_index(['prediction_id'])\
                          .groupby(level=0).apply(shortened_site_names)\
                          .reset_index()\
                          .rename(columns={0:'site_names'})


def annotate_site_names(breakpoint, breakpoint_site):
    """ Add `site_names` column based on presence absence of breakpoints
    """

    breakpoint_site_names = calculate_site_names(breakpoint_site)

    breakpoint = breakpoint.merge(breakpoint_site_names)

    return breakpoint


def annotate_tumour_count_max(breakpoints):
    """ Annotation of max tumour count across tumour samples
    """

    counts_columns = get_counts_columns(breakpoints)

    # Max tumour count across sites
    breakpoints.set_index('prediction_id', inplace=True)
    breakpoints['tumour_count_max'] = breakpoints[counts_columns].max(axis='columns')
    breakpoints.reset_index(inplace=True)

    # Abbreviated site name group
    breakpoints = annotate_site_names(breakpoints)

    return breakpoints


def filter_breakpoints(breakpoints):
    """ Filtering germline and putative false positives
    """

    # No germline
    breakpoints = breakpoints[breakpoints['normal_blood_count'] == 0]

    # No common rearrangements
    breakpoints = breakpoints[breakpoints['common'] <= max_other_libraries]

    # Breakpoint probability filter
    breakpoints = breakpoints[breakpoints['align_prob'] > align_prob_score_min]
    breakpoints = breakpoints[breakpoints['valid_prob'] > valid_prob_score_min]
    breakpoints = breakpoints[breakpoints['chimeric_prob'] > chimeric_prob_score_min]

    # Filter small deletions
    breakpoints = breakpoints[(breakpoints['type'] != 'deletion') | (breakpoints['break_distance'] > 1000)]

    return breakpoints


def stack_counts(breakpoints):
    """ Convert to stacked counts table
    """


    counts_columns = get_counts_columns(breakpoints)
    break_counts = breakpoints.set_index('prediction_id')[counts_columns].stack().reset_index()
    break_counts.columns = ['prediction_id', 'site_id', 'num_span']
    break_counts['site_id'] = break_counts['site_id'].apply(lambda a: a[:-len('_count')])
    breakpoints = breakpoints.drop(counts_columns + ['normal_blood_count'], axis=1)
    breakpoints = breakpoints.merge(break_counts, left_on='prediction_id', right_on='prediction_id')

    return breakpoints


def stack_sides(breakpoints):
    """ Convert to stacked cluster end 1 and 2 table
    """

    breakpoints_common = breakpoints.filter(regex='[^12]$')

    def remove_side_suffix(column):
        if column.endswith('_1') or column.endswith('_2'):
            return column[:-2]
        elif column.endswith('1') or column.endswith('2'):
            return column[:-1]
        return column

    breakpoints_side_1 = breakpoints.set_index('prediction_id').filter(regex='1$', axis=1).rename(columns=remove_side_suffix).reset_index().drop_duplicates()
    breakpoints_side_1['cluster_end'] = 0
    
    breakpoints_side_2 = breakpoints.set_index('prediction_id').filter(regex='2$', axis=1).rename(columns=remove_side_suffix).reset_index().drop_duplicates()
    breakpoints_side_2['cluster_end'] = 1

    breakpoints_sides = pd.concat([breakpoints_side_1, breakpoints_side_2], ignore_index=True)

    breakpoints = breakpoints_common.merge(breakpoints_sides, left_on='prediction_id', right_on='prediction_id')

    breakpoints = breakpoints.set_index(['prediction_id', 'cluster_end']).reset_index()

    return breakpoints


def annotate_high_confidence(breakpoints):
    """ Annotation of high confident breakpoints
    """

    # High confidence breakpoints
    breakpoints['high_conf'] = ((breakpoints['tumour_count_max'] >= high_conf_min_tumour_count_max) &
                                (breakpoints['num_split'] >= high_conf_min_num_split)) * 1

    return breakpoints



def add_annotations(breakpoints):

    breakpoints = annotate_tumour_count_max(breakpoints)
    breakpoints = annotate_site_names(breakpoints)

    breakpoints = filter_breakpoints(breakpoints)

    breakpoints = stack_counts(breakpoints)
    breakpoints = stack_sides(breakpoints)

    breakpoints = annotate_high_confidence(breakpoints)

    breakpoints = breakpoints.sort(['prediction_id', 'site_id'])

    return breakpoints


def annotate_expression_correlation(breakpoints, expression):
    """ Add presence correlation with expression expr_corr column
    """

    breakpoints = breakpoints.merge(expression)

    breakpoints.set_index(['prediction_id', 'cluster_end'], inplace=True)

    breakpoints['presences'] = (breakpoints['num_span'] > 0) * 1
    
    expr_corr = breakpoints[['presences', 'recentred_expression']].groupby(level=[0, 1]).corr().fillna(0.0)
    expr_corr = expr_corr.xs('presences', level=2)['recentred_expression']
    
    breakpoints['expr_corr'] = expr_corr

    breakpoints.reset_index(inplace=True)
    
    breakpoints = breakpoints.drop('presences', axis=1)

    return breakpoints


def annotate_size_class(data):
    """ Add the categorical column size_class.

    Expects columns:
        - position_1
        - position_2
        - type
    """

    length = (data['position_1'] - data['position_2']).abs()
    data['size_class'] = pd.cut(
        length,
        [0, 1e4, 1e6, 1e8, 1e10],
        labels=['0-1K', '1K-1M', '1-100M', '>100M'])
    data['size_class'] = data['size_class'].astype(str)
    data.loc[data['type'] == 'translocation', 'size_class'] = 'Tr'
    data['size_class'] = pd.Categorical(
        data['size_class'],
        categories=reversed(['0-1K', '1K-1M', '1-100M', '>100M', 'Tr']))

    return data


