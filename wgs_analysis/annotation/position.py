'''
Position annotation functionality

Annotations for positional data such as SNVs and breakends.

Functions take pandas tables and add additional columns.
'''

import collections
import numpy as np
import pandas as pd

from wgs_analysis.algorithms.position import calculate_adjacent_density
from wgs_analysis.algorithms.position import calculate_adjacent_distance


def annotate_adjacent_density(df, stddev=5000):
    """ Add adjacent_density column

    Args:
        df(pandas.DataFrame): positions table

    Returns:
        pandas.DataFrame: original table with `adjacent_density` column

    Calculate the density of adjacent positions at each position using a gaussian
    kernal with standard deviation stddev.

    """

    positions = (
        df[['chrom', 'coord']]
        .drop_duplicates()
        .set_index(['chrom', 'coord'], drop=False)
        .sort_index())
    positions['coord'] = positions['coord'].astype(float)

    adjacent_density = (
        positions
        .groupby(level='chrom')['coord']
        .transform(calculate_adjacent_density, stddev))

    df.set_index(['chrom', 'coord'], inplace=True)
    df['adjacent_density'] = adjacent_density
    df.reset_index(inplace=True)

    return df


def annotate_adjacent_distance(df):
    """ Add adjacent_distance column

    Args:
        df(pandas.DataFrame): positions table

    Returns:
        pandas.DataFrame: original table with `adjacent_distance` column

    Calculate the closest distance to an adjacent position.
    """

    positions = (
        df[['chrom', 'coord']]
        .drop_duplicates()
        .set_index(['chrom', 'coord'], drop=False)
        .sort_index())
    positions['coord'] = positions['coord'].astype(float)

    adjacent_distance = (
        positions
        .groupby(level='chrom')['coord']
        .transform(lambda pos: calculate_adjacent_distance(pos.values)))

    df.set_index(['chrom', 'coord'], inplace=True)
    df['adjacent_distance'] = adjacent_distance
    df.reset_index(inplace=True)

    return df

