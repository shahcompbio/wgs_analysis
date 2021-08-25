import pandas as pd
import numpy as np


def position_segment_merge(positions, segments):
    """
    Merge positions with segments that contain them
    
    Args:
        positions (pandas.DataFrame): ['chrom', 'coord'] columns required
        segments (pandas.DataFrame): ['chrom', 'start', 'end'] columns required

    Returns:
        pandas.DataFrame: merged table with ['chrom', 'coord', 'start', 'end'] columns

    Assuming a set of non-overlapping segments, merge a set of positions so that
    each entry in the new table provides coord and containing segment start/end
    """

    # Check if any segments overlap
    for chromosome, df in segments.groupby('chrom'):
        if (df['start'].values[1:] < df['end'][:-1]).any():
            raise Exception('overlapping segments')

    positions = positions[['chrom', 'coord']].copy()
    segments = segments[['chrom', 'start', 'end']].copy()
    
    segments['chrom_1'] = segments['chrom']
    segments['coord'] = segments['start']
    
    positions['is_pos'] = 1
    segments['is_pos'] = 0
    
    merged = pd.concat([positions, segments], ignore_index=True)\
               .sort_values(['chrom', 'coord', 'is_pos'])\
               .drop_duplicates()
    
    merged['chrom_1'] = merged['chrom_1'].fillna(method='ffill')
    merged['start'] = merged['start'].fillna(method='ffill')
    merged['end'] = merged['end'].fillna(method='ffill')
    
    merged = merged[merged['is_pos'] == 1]
    
    merged = merged[(merged['chrom'] == merged['chrom_1']) &
                    (merged['coord'] >= merged['start']) &
                    (merged['coord'] <= merged['end'])]

    merged = merged.drop(['is_pos', 'chrom_1'], axis=1)
    merged['start'] = merged['start'].astype(int)
    merged['end'] = merged['end'].astype(int)

    return merged


def vrange(starts, lengths):
    """ Create concatenated ranges of integers for multiple start/length

    Args:
        starts (numpy.array): starts for each range
        lengths (numpy.array): lengths for each range (same length as starts)

    Returns:
        numpy.array: concatenated ranges

    See the following illustrative example:

        starts = np.array([1, 3, 4, 6])
        lengths = np.array([0, 2, 3, 0])

        print vrange(starts, lengths)
        >>> [3 4 4 5 6]

    """
    
    # Repeat start position index length times and concatenate
    cat_start = np.repeat(starts, lengths)

    # Create group counter that resets for each start/length
    cat_counter = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)

    # Add group counter to group specific starts
    cat_range = cat_start + cat_counter

    return cat_range


def interval_position_overlap(intervals, positions):
    """ Map intervals to contained positions

    Args:
        intervals (numpy.array): start and end of intervals with shape (N,2) for N intervals
        positions (numpy.array): positions, length M, must be sorted

    Returns:
        numpy.array: interval index, length L (arbitrary)
        numpy.array: position index, length L (same as interval index)

    Given a set of possibly overlapping intervals, create a mapping of positions that are contained
    within those intervals.

    """

    # Search for start and end of each interval in list of positions
    start_pos_idx = np.searchsorted(positions, intervals[:,0])
    end_pos_idx = np.searchsorted(positions, intervals[:,1])

    # Calculate number of positions for each segment
    lengths = end_pos_idx - start_pos_idx

    # Interval index for mapping
    interval_idx = np.repeat(np.arange(len(lengths)), lengths)

    # Position index for mapping 
    position_idx = vrange(start_pos_idx, lengths)

    return interval_idx, position_idx


def interval_position_overlap_unsorted(intervals, positions):
    """ Map intervals to contained positions

    Args:
        intervals (numpy.array): start and end of intervals with shape (N,2) for N intervals
        positions (numpy.array): positions, length M

    Returns:
        numpy.array: interval index, length L (arbitrary)
        numpy.array: position index, length L (same as interval index)

    Given a set of possibly overlapping intervals, create a mapping of positions that are contained
    within those intervals.

    """

    pos_sort_idx = np.argsort(positions)
    rev_idx = np.arange(len(pos_sort_idx), dtype=int)[pos_sort_idx]

    interval_idx, position_idx = interval_position_overlap(intervals, positions[pos_sort_idx])

    return interval_idx, rev_idx[position_idx]

