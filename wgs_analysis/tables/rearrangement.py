'''
# Rearrangement table

Subsetting of rearrangements and generation of derivative tables.

'''

import pandas as pd

import ith_project.analysis.params as params
import ith_project.analysis.annotation as annotation
import ith_project.analysis.annotation.rearrangement

def expression_correlated(breakpoints, expression):

    breakpoints = annotation.rearrangement.annotate_expression_correlation(breakpoints, expression)

    selected_cluster_ids = breakpoints.loc[(breakpoints['expr_corr'] >= params.expression_correlation_threshold) |
                                           (breakpoints['expr_corr'] <= -params.expression_correlation_threshold), 'cluster_id']

    breakpoints = breakpoints[breakpoints['cluster_id'].isin(selected_cluster_ids.values)]

    return breakpoints

def get_brkend(brk, side, data_cols):
    cols = ['chromosome', 'strand', 'position']
    side_cols = [a+'_'+side for a in cols]
    brkend = brk[['prediction_id']+side_cols+data_cols]
    brkend = brkend.rename(columns=dict(zip(side_cols, cols)))
    brkend['side'] = side
    return brkend

def get_brkends(brk, data_cols):
    brkends = pd.concat([get_brkend(brk, '1', data_cols),
                         get_brkend(brk, '2', data_cols)], ignore_index=True)
    return brkends
