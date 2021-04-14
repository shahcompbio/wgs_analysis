import bisect
import collections
import pandas as pd
import numpy as np


def create_breakends(breakpoints, data_cols=(), id_col='prediction_id'):
    breakends = breakpoints[[id_col, 'chromosome_1', 'strand_1', 'position_1', 'chromosome_2', 'strand_2', 'position_2']].copy()
    breakends.set_index(id_col, inplace=True)
    breakends.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in breakends.columns])
    breakends = breakends.stack()
    breakends.index.names = (id_col, 'prediction_side')
    breakends = breakends.reset_index()
    breakends['prediction_side'] = np.where(breakends['prediction_side'] == '1', 0, 1)
    breakends = breakends.merge(breakpoints[[id_col] + list(data_cols)], on=id_col)
    return breakends


class BreakpointDatabase(object):
    def __init__(self, breakpoints, id_col='prediction_id'):
        self.breakends = (
            create_breakends(breakpoints, id_col=id_col)
            .sort_values(['chromosome', 'strand', 'position'])
            .set_index(['chromosome', 'strand']))

    def query(self, row, extend=0):
        side_matches = {}

        for side in ('1', '2'):
            chromosome = row['chromosome_' + side]
            strand = row['strand_' + side]
            position = row['position_' + side]

            idx1 = self.breakends.xs((chromosome, strand))['position'].searchsorted(position - extend)
            idx2 = self.breakends.xs((chromosome, strand))['position'].searchsorted(position + extend, side='right')

            side_matches[side] = self.breakends.xs((chromosome, strand)).iloc[idx1:idx2]

        matches = pd.merge(
            side_matches['1'],
            side_matches['2'],
            on=['prediction_id'],
            suffixes=('_1', '_2'),
        )

        matches = matches.query('prediction_side_1 != prediction_side_2')

        return set(matches['prediction_id'].values)


def match_breakpoints(reference_breakpoints, target_breakpoints, window_size=500):
    """ Match similar target breakpoints to a set of reference breakpoints  
    """

    reference_db = BreakpointDatabase(reference_breakpoints)

    match_data = []

    for idx, row in target_breakpoints.iterrows():
        for reference_prediction_id in reference_db.query(row, extend=window_size):
            match_data.append({
                'target_prediction_id': row['prediction_id'],
                'reference_prediction_id': reference_prediction_id,
            })

    match_data = pd.DataFrame(match_data)

    return match_data
