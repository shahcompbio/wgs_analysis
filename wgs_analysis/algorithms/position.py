import collections
import numpy as np
from scipy import stats


def calculate_smoothed(position, adjacent, stddev):
    return np.sum(stats.norm(loc=position, scale=stddev).pdf(adjacent))


def calculate_adjacent_density(positions, stddev):
    """ Smoothed position density

    Args:
        positions(numpy.array) : list of positions
        stddev(float) : standard deviation for gaussian smoothing

    Calculate the kernal density of neighbouring positions for each
    position.  Use a gaussian kernal with given standard deviation
    """

    bandwidth = 3.0 * stddev
    left_bandwidth_positions = collections.deque()
    left_counts = list()
    for position in positions.values:
        while len(left_bandwidth_positions) > 0 and left_bandwidth_positions[0] < position - bandwidth:
            left_bandwidth_positions.popleft()
        left_counts.append(calculate_smoothed(position, left_bandwidth_positions, stddev))
        left_bandwidth_positions.append(position)
    right_bandwidth_positions = collections.deque()
    right_counts = list()
    for position in reversed(positions.values):
        while len(right_bandwidth_positions) > 0 and right_bandwidth_positions[0] > position + bandwidth:
            right_bandwidth_positions.popleft()
        right_counts.append(calculate_smoothed(position, right_bandwidth_positions, stddev))
        right_bandwidth_positions.append(position)
    right_counts = list(reversed(right_counts))
    return np.array(left_counts) + np.array(right_counts)


def calculate_adjacent_distance(positions):
    """ Distance to nearest adjacent.

    Args:
        positions(numpy.array) : list of positions

    Returns:
        distances(numpy.array): list of distances

    """
    dist_adj = np.absolute(positions[:-1] - positions[1:])

    dist_left = np.ones(positions.shape) * np.inf
    dist_left[1:] = dist_adj

    dist_right = np.ones(positions.shape) * np.inf
    dist_right[:-1] = dist_adj

    dist = np.minimum(dist_left, dist_right)
    dist[dist == np.inf] = np.nan

    return dist
