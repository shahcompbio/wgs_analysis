'''
# Rearrangement table

Subsetting of rearrangements and generation of derivative tables.

'''

import pandas as pd


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
