'''
Position annotation functionality

Annotations for positional data such as SNVs and breakends.

Functions take pandas tables and add additional columns.
'''

import collections
import numpy as np
import pandas as pd

from wgs_analysis.algorithms.position import calculate_adjacent_density


def annotate_adjacent_density(df, stddev=5000):
    """ Add adjacent_density column

    Args:
        df(pandas.DataFrame): positions table

    Returns:
        pandas.DataFrame: original table with `adjacent_density` column

    Calculate the density of adjacent positions at each position using a gaussian
    kernal with standard deviation stddev.

    """

    positions = df[['chromosome', 'position']].drop_duplicates()\
                                      .set_index(['chromosome', 'position'], drop=False)\
                                      .sort_index()
    positions['position'] = positions['position'].astype(float)

    adjacent_density = positions.groupby('chromosome')['position']\
                                .transform(calculate_adjacent_density, stddev)

    df.set_index(['chromosome', 'position'], inplace=True)
    df['adjacent_density'] = adjacent_density
    df.reset_index(inplace=True)

    return df

