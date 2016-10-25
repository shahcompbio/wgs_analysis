import pandas as pd

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

    positions = positions[['chrom', 'coord']].copy()
    segments = segments[['chrom', 'start', 'end']].copy()
    
    segments['chrom_1'] = segments['chrom']
    segments['coord'] = segments['start']
    
    positions['is_pos'] = 1
    segments['is_pos'] = 0
    
    merged = pd.concat([positions, segments], ignore_index=True)\
               .sort(['chrom', 'coord', 'is_pos'])\
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
