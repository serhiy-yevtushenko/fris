"""Various utility functions that could be used for analysing nearest neighbours classifiers."""

import math
from collections.abc import Callable
from collections.abc import Sequence
from typing import cast
from typing import Final
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from matplotlib.figure import Figure
from scipy.spatial import KDTree
from scipy.special import comb
from sklearn.neighbors import NearestNeighbors

from fris.fris_types import ClassType
from fris.fris_types import DataPoint
from fris.fris_types import DataPointArray

TREE_ELEMENTS_INDICES: Final[int] = 1


class NearestNeighbourSearcher(NearestNeighbors):
    """Helper classes for finding indices of the nearest neighbours."""

    def __init__(self, X: Optional[DataPointArray] = None) -> None:
        super().__init__()
        self.X = X
        if X is not None:
            self.fit(X)

    def get_indexes_of_nearest_neighbours(self, x: DataPoint, neighbor_count: int) -> list[int]:
        # TODO: Adapt work on neighbour_count=1
        return cast(list[int], self.kneighbors(X=np.array([x]), n_neighbors=neighbor_count, return_distance=False)[0])


class KDTreeNearestNeighbourSearcher(NearestNeighbourSearcher):
    """KDTree based implementation of the NearestNeighbourSearcher."""

    def __init__(self, x: DataPointArray) -> None:
        super().__init__()
        self.xkd = KDTree(x)

    def get_indexes_of_nearest_neighbours(self, x: DataPoint, neighbor_count: int) -> list[int]:
        if neighbor_count == 1:
            return [self.xkd.query(x, neighbor_count)[TREE_ELEMENTS_INDICES]]
        return cast(list[int], self.xkd.query(x, neighbor_count)[TREE_ELEMENTS_INDICES])


def accuracy_loss(y_mth: ClassType | float, y_actual: ClassType | float) -> float:
    return y_mth != y_actual


def abs_loss(y_mth: ClassType | float, y_actual: ClassType | float) -> float:
    assert isinstance(y_mth, float)
    assert isinstance(y_actual, float)
    return abs(y_mth - y_actual)


def squared_loss(y_mth: ClassType | float, y_actual: ClassType | float) -> float:
    assert isinstance(y_mth, float)
    assert isinstance(y_actual, float)
    return (y_mth - y_actual) ** 2


# this one is suitable for accuracy and mean absolute error
def average_loss(summary_loss: float, dataset_size: int) -> float:
    return summary_loss / dataset_size


# this loss aggregator is intended for rmse
def squared_average_loss(summary_loss: float, dataset_size: int) -> float:
    average_loss = summary_loss / dataset_size
    return math.sqrt(average_loss)


def compactness_profile(
    X: DataPointArray,
    y: Sequence[ClassType | float],
    loss_function: Callable[[float | ClassType, float | ClassType], float] = accuracy_loss,
    loss_aggregator: Callable[[float, int], float] = average_loss,
) -> list[float]:
    """Compute compactness profile, as described by Vorontsov and Co."""
    result = []
    neighbour_searcher = NearestNeighbourSearcher(X)
    ordered_neighbours = []

    for i in range(len(X)):
        # Taking indices of elements and storing starting from second, as first
        # element index will be same, as current element (if all elements in set X
        # have distinct location

        neighbours = neighbour_searcher.get_indexes_of_nearest_neighbours(X[i], neighbor_count=len(X))[1:]
        # print(f"\n neighbours indexes for {X[i]} {len(X)} :{neighbours}")
        # print(i, [X[j] for j in neighbours])
        ordered_neighbours.append(neighbours)

    for m in range(len(X) - 1):
        # print(m)
        m_th_sum = 0.0
        for i in range(len(X)):
            m_th_neighbour_index = ordered_neighbours[i][m]
            # print(i, m_th_neighbour_index)
            m_th_sum += loss_function(y[m_th_neighbour_index], y[i])

        m_th_sum = loss_aggregator(m_th_sum, len(X))
        # print("m_th_sum", m_th_sum)
        result.append(m_th_sum)
    return result


DEFAULT_X_LABEL_TITLE: Final[str] = "k"
DEFAULT_Y_LABEL_TITLE: Final[str] = "% of objects of other classes \n among k ordered neighbours"


def compactness_profile_figure(
    profile: list[float],
    x_label: Optional[str] = DEFAULT_X_LABEL_TITLE,
    y_label: Optional[str] = DEFAULT_Y_LABEL_TITLE,
    color: Union[str, Tuple[int, int, int]] = "b",
) -> Figure:
    """Create matplotlib Figure for the compactness profile."""
    figure = Figure()
    ax = figure.add_subplot(111)
    ax.set_xticks([])
    ax.set_yticks([0.1 * i for i in range(11)])
    ax.set_ylim((-0.1, 1.1))

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    ax.set_xlim((1, len(profile)))
    ax.plot([x + 1 for x in range(len(profile))], profile, color)
    return figure


def binomial_coefficient(n: int, k: int) -> float:
    return cast(float, comb(n, k))


def gamma_ccv(training_set_size: int, dataset_len: int, j: int) -> float:
    """Compute function gamma for computing CCV (complete cross validation value) with help of compactness profile.

    :param training_set_size - size of the training set
    :param dataset_len - size of the whole dataset
    :param j - step for the CCV
    """
    nominator = training_set_size
    # The biggest value of i is training_set_size-2
    for i in range(training_set_size - 1):
        nominator *= dataset_len - 1 - j - i
    demoninator = 1
    # The biggest value of i is training_set_size-1
    for i in range(training_set_size):
        demoninator *= dataset_len - 1 - i
    result = nominator / demoninator
    return result


def complete_cross_validation(X: DataPointArray, y: list[ClassType], test_set_size: int) -> float:
    profile = compactness_profile(X, y)
    return compute_ccv(profile, test_set_size)


def compute_ccv(profile: list[float], test_set_size: int) -> float:
    dataset_len = len(profile) + 1
    training_set_size = dataset_len - test_set_size
    result = 0.0
    for j in range(1, test_set_size + 1):
        result += profile[j - 1] * gamma_ccv(training_set_size, dataset_len, j)
    return result


def compute_ccv_for_training_set_sizes(profile: list[float], training_set_sizes: list[int]) -> dict[int, float]:
    dataset_len = len(profile) + 1
    results = {}
    for training_set_size in training_set_sizes:
        assert (
            1 <= training_set_size < dataset_len
        ), f"Bad training set size {training_set_size} dataset_length {dataset_len}"
        test_set_size = dataset_len - training_set_size
        results[training_set_size] = compute_ccv(profile, test_set_size)
    return results


def compute_ccvs(profile: list[float], steps: int) -> dict[int, float]:
    """Compute complete cross-validation values using given `profile` for steps points.

    @param profile - compactness profile for the dataset
    @param steps - number of points, on which complete cross validation values should be computed.
                   The value of steps is evenly distributed between 1 and length of profile.
    @return Dictionary of CCVs for training set sizes, computed using steps from 1 to profile len in equal distances.
            The key of dictionary is the size of the training set, value - complete cross validation error for this
            dataset for a given training set size.
    """
    steps = min(steps, len(profile))
    min_training_size = 1
    if steps == 1:
        training_set_sizes = [min_training_size]
    else:
        max_training_size = len(profile)  # (Dataset len-1)
        delta = (max_training_size - min_training_size) / (steps - 1)
        if delta < 1.0:
            raise ValueError(f"Too many steps {steps} are requested for dataset of size {len(profile)+1}")
        training_set_sizes = [min_training_size + int(round(delta * j)) for j in range(0, steps)]
    return compute_ccv_for_training_set_sizes(profile, training_set_sizes)
