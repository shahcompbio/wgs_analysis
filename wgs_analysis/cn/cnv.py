import os
import pandas as pd
import numpy as np

def load_cn_data(fpath, method='WGS-REMIXT-POSTPROCESS'):
    """ Load CN data and label `is_valid`
    - fpath: file path to CN data
    - method: defaults to "WGS-REMIXT-POSTPROCESS"
    
    Returns:
    - remixt_cn-like pd.DataFrame with `is_valid` column
    """
    assert os.path.exists(fpath), f'ERROR: {fpath} does not exist'
    cn = pd.DataFrame()
    if method == 'WGS-REMIXT-POSTPROCESS':
        cn = pd.read_table(fpath, dtype={'chromosome':str})
        cn['total_raw'] = cn['major_raw'] + cn['minor_raw']
        cn['segment_length'] = cn['end'] - cn['start']
        cn['length_ratio'] = cn['length'] / cn['segment_length']
        cn['is_valid'] = ( # heuristic soft filter
            (cn['length'] > 100000) &
            (cn['minor_readcount'] > 100) &
            (cn['length_ratio'] > 0.8)
        )
    else:
        raise ValueError(f'ERROR: method={method}')
    return cn

def label_cnv(cn, select_valid=True):
    """ Label CNV events based on some heuristics
    - cn: CN pd.DataFrame
    - select_valid: only calculate CNV based on `is_valid` (recommended)
    
    Returns: 
    - same cn DataFrame but with events labelled
    """
    valid = cn[cn['is_valid']].copy()
    ploidy = valid['total_raw'].mean()
    cn['log_change'] = np.log2(cn['total_raw']) - np.log2(ploidy)
    gain = cn['log_change'] >= 0.5
    loss = cn['log_change'] < -0.5
    cn['event'] = gain + -1 * loss # never overlapping
    return cn