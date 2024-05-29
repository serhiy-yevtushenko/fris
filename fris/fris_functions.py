"""Various function used in the implementation of FRiS-Based algorithms."""

import logging
from functools import reduce
from typing import cast
from typing import Collection

import numpy as np

from fris.fris_types import CachedDistanceMatrix
from fris.fris_types import DataPoint
from fris.fris_types import DistanceFunctionType


def fris_function(r_1: float, r_2: float) -> float:
    assert r_1 >= 0, f"Distance {r_1} should be non-negative"
    assert r_2 >= 0, f"Distance {r_2} should be non-negative"
    if r_2 == 0 and r_1 == 0:
        # This is an experimental completion of function for case
        # points, that are same, but belong to different classes
        return 0.0
    return (r_2 - r_1) / (r_2 + r_1)


def fast_euclidean_distance(x: DataPoint, y: DataPoint) -> float:
    """Distance used in FRiS-based algorithms - simple L2 distance.

    This function works almost two times faster in comparison to euclidean_distance from scipy.
    """
    return cast(float, np.linalg.norm(x - y))


def rival_similarity(
    z: DataPoint, a: DataPoint, b: DataPoint, distance: DistanceFunctionType = fast_euclidean_distance
) -> float:
    """Compute function of rival similarity (FRiS).

    Computes the rival similarity of object z in
    competition between objects a and b
    This is denoted in descriptions of FRiS algorithms
    as $F(z, a|b)$.
    The range of values of function lays between -1 and 1.
    -1 means that z is equal to b
    1 means that z is equal to a
    0 means that z is equidistant to a and b.

    Parameters
    ----------
    @param: distance - function used to compute distance between data points
    """
    d_2 = distance(z, b)
    d_1 = distance(z, a)
    return fris_function(d_1, d_2)


def rival_similarity_index(z_index: int, a_index: int, b_index: int, distance_matrix: CachedDistanceMatrix) -> float:
    """Compute function of rival similarity, using indexes of data points and cached distance matrix.

    Computes the rival similarity of object z in
    competition between objects a and b
    This is denoted in descriptions of FRiS algorithms
    as $F(z, a|b)$.
    The range of values of function lays between -1 and 1.
    -1 means that z is equal to b
    1 means that z is equal to a
    0 means that z is equidistant to a and b.
    """
    r_b = distance_matrix[z_index, b_index]
    r_a = distance_matrix[z_index, a_index]
    return fris_function(r_a, r_b)


def rival_similarity_reduced(
    u: DataPoint, x: DataPoint, r_star: float, distance: DistanceFunctionType = fast_euclidean_distance
) -> float:
    """Compute function of rival similarity (clustering based variant).

    Computes the rival similarity of object u in
    competition between objects x and virtual object, distance to
    which is equal to r_*
    The range of values of function lays between -1 and 1.
    -1 means that z is equal to b
    1 means that z is equal to a
    0 means that z is equidistant to a and b.
    Parameters
    ----------
    :param u - first object from dataset
    :param x - other object from dataset
    :param r_star - distance to virtual object. Usually positive number
    :param distance - function used for calculating distance
    """

    # print "rival_similarity_reduced", u, x, r_star
    d2 = distance(u, x)
    return fris_function(d2, r_star)


def average(compactness_of_classes: Collection[float]) -> float:
    if len(compactness_of_classes) == 0:
        return 0
    return sum(compactness_of_classes) / len(compactness_of_classes)


def geometric_average(compactness_of_classes: Collection[float]) -> float:
    product = reduce(lambda a, b: a * b, compactness_of_classes, 1.0)
    if len(compactness_of_classes) > 0:
        # in some extreme cases product could be negative
        # therefore, it makes sense to prevent error by taking abs of it
        if product < 0:
            logging.warning(
                f"Product {product=} is smaller then zero.\n"
                f"\t{compactness_of_classes=} {len(compactness_of_classes)=}"
            )
        return cast(float, abs(product) ** (1 / len(compactness_of_classes)))
    return product
