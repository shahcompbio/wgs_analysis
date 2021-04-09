import bisect
import collections
import pandas as pd


class BreakpointDatabase(object):
    def __init__(self, breakpoints, id_col='prediction_id'):
        self.positions = collections.defaultdict(list)
        self.break_ids = collections.defaultdict(set)

        for idx, row in breakpoints.iterrows():
            for side in ('1', '2'):
                self.positions[(row['chromosome_' + side], row['strand_' + side])].append(row['position_' + side])
                self.break_ids[(row['chromosome_' + side], row['strand_' + side], row['position_' + side])].add(
                    (row[id_col], side))

        for key in self.positions.keys():
            self.positions[key] = sorted(self.positions[key])

    def query(self, row, extend=0):
        matched_ids = list()

        for side in ('1', '2'):
            chrom_strand_positions = self.positions[(row['chromosome_' + side], row['strand_' + side])]
            idx = bisect.bisect_left(chrom_strand_positions, row['position_' + side] - extend)
            side_matched_ids = list()

            while idx < len(chrom_strand_positions):
                pos = chrom_strand_positions[idx]
                dist = abs(pos - row['position_' + side])

                if pos >= row['position_' + side] - extend and pos <= row['position_' + side] + extend:
                    for break_id in self.break_ids[(row['chromosome_' + side], row['strand_' + side], pos)]:
                        side_matched_ids.append((break_id, dist))

                if pos > row['position_' + side] + extend:
                    break
                idx += 1
            matched_ids.append(side_matched_ids)

        matched_ids_bypos = list()
        for matched_id_1, dist_1 in matched_ids[0]:
            for matched_id_2, dist_2 in matched_ids[1]:
                if matched_id_1[0] == matched_id_2[0] and matched_id_1[1] != matched_id_2[1]:
                    matched_ids_bypos.append((dist_1 + dist_2, matched_id_1[0]))

        ids = set([v[1] for v in matched_ids_bypos])
        return sorted(ids)


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
