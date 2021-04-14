import unittest
import numpy as np
import pandas as pd

import wgs_analysis.algorithms.rearrangement


def generate_random_breakpoints(num_breakpoints, max_position=10000, seed=123):
    np.random.seed(seed)

    chromosomes = ['1', '2']
    strands = ['+', '-']

    breakpoints = pd.DataFrame({
        'chromosome_1': np.random.choice(chromosomes, size=num_breakpoints),
        'chromosome_2': np.random.choice(chromosomes, size=num_breakpoints),
        'strand_1': np.random.choice(strands, size=num_breakpoints),
        'strand_2': np.random.choice(strands, size=num_breakpoints),
        'position_1': np.random.randint(max_position, size=num_breakpoints),
        'position_2': np.random.randint(max_position, size=num_breakpoints),
    })
    breakpoints['prediction_id'] = range(len(breakpoints.index))

    return breakpoints


def find_matches_brute(breakpoints_1, breakpoints_2, extend=500):
    matches = []

    for idx1, row1 in breakpoints_1.iterrows():
        for idx2, row2 in breakpoints_2.iterrows():

            match1 = (
                (row1['chromosome_1'] == row2['chromosome_1']) &
                (row1['strand_1'] == row2['strand_1']) &
                (abs(row1['position_1'] - row2['position_1']) <= extend) &
                (row1['chromosome_2'] == row2['chromosome_2']) &
                (row1['strand_2'] == row2['strand_2']) &
                (abs(row1['position_2'] - row2['position_2']) <= extend))
            match2 = (
                (row1['chromosome_1'] == row2['chromosome_2']) &
                (row1['strand_1'] == row2['strand_2']) &
                (abs(row1['position_1'] - row2['position_2']) <= extend) &
                (row1['chromosome_2'] == row2['chromosome_1']) &
                (row1['strand_2'] == row2['strand_1']) &
                (abs(row1['position_2'] - row2['position_1']) <= extend))

            if match1 or match2:
                matches.append({
                    'from_idx': row1['prediction_id'],
                    'to_idx': row2['prediction_id'],
                })

    matches = pd.DataFrame(matches)

    return matches


class BreakpointDatabaseTestCase(unittest.TestCase):
    def test_find_matching(self):
        extend = 500

        random_breakpoints = generate_random_breakpoints(1000, max_position=2000, seed=1)

        breakpoint_db = wgs_analysis.algorithms.rearrangement.BreakpointDatabase(random_breakpoints)

        test_breakpoints = generate_random_breakpoints(1000, max_position=2000, seed=2)

        matches = []

        for i, (idx, row) in enumerate(test_breakpoints.iterrows()):
            for match in breakpoint_db.query(row, extend=extend):
                matches.append({
                    'from_idx': match,
                    'to_idx': row['prediction_id'],
                })

        matches = pd.DataFrame(matches)

        test_matches = find_matches_brute(random_breakpoints, test_breakpoints, extend=extend)

        test_results = pd.merge(
            matches.assign(matches=1),
            test_matches.assign(test_matches=1),
            how='outer',
        )

        self.assertFalse(test_results['matches'].isnull().any())
        self.assertFalse(test_results['test_matches'].isnull().any())

