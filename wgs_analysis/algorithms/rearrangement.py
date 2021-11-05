import bisect
import collections
import pandas as pd
import numpy as np
import scipy.sparse


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
        self.id_col = id_col
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

            if (chromosome, strand) not in self.breakends.index:
                return set()

            idx1 = self.breakends.xs((chromosome, strand))['position'].searchsorted(position - extend)
            idx2 = self.breakends.xs((chromosome, strand))['position'].searchsorted(position + extend, side='right')

            side_matches[side] = self.breakends.xs((chromosome, strand)).iloc[idx1:idx2]

        matches = pd.merge(
            side_matches['1'],
            side_matches['2'],
            on=[self.id_col],
            suffixes=('_1', '_2'),
        )

        matches = matches.query('prediction_side_1 != prediction_side_2')

        return set(matches[self.id_col].values)


def match_breakpoints(reference_breakpoints, target_breakpoints, id_col='prediction_id', window_size=500):
    """ Match similar target breakpoints to a set of reference breakpoints  
    """

    reference_db = BreakpointDatabase(reference_breakpoints, id_col=id_col)

    match_data = []

    for idx, row in target_breakpoints.iterrows():
        for reference_id in reference_db.query(row, extend=window_size):
            match_data.append({
                'target_id': row[id_col],
                'reference_id': reference_id,
            })

    match_data = pd.DataFrame(match_data, columns=['target_id', 'reference_id'])

    return match_data


def identify_matched_breakpoints(breakpoints, window_size=500):
    """ Identify common breakpoints.
    
    Args:
        breakpoints (DataFrame): breakpoint data
        
    Returns:
        DataFrame: breakpoints with component_id
        
    The component_id column will have the same value for similar breakpoints
    """

    breakpoints = breakpoints.copy()

    breakpoints['idx'] = range(len(breakpoints.index))

    self_matches = match_breakpoints(
        breakpoints, breakpoints, id_col='idx', window_size=window_size)

    row = self_matches['reference_id'].values
    col = self_matches['target_id'].values
    data = np.ones(len(self_matches.index))
    max_index = self_matches[['reference_id', 'target_id']].max().max()

    assert max_index == breakpoints['idx'].max()

    matrix = scipy.sparse.csr_matrix((data, (row, col)), shape=(max_index + 1, max_index + 1))

    num_components, component_ids = scipy.sparse.csgraph.connected_components(
        matrix, directed=False, return_labels=True)

    components = pd.DataFrame({
        'idx': range(len(component_ids)),
        'component_id': component_ids,
    })

    breakpoints = breakpoints.merge(components, how='outer')

    assert not breakpoints['component_id'].isnull().any()
    assert not breakpoints['idx'].isnull().any()
    
    return breakpoints


