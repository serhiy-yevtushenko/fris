# -*- coding: utf-8 -*-
"""Algorithms, data structures and helper functions for the implementation of FRiS-Stolp algorithm."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable
from typing import cast
from typing import Final
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from numpy._typing import NDArray
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from fris.fris_functions import average
from fris.fris_functions import fast_euclidean_distance
from fris.fris_functions import fris_function
from fris.fris_functions import rival_similarity
from fris.fris_functions import rival_similarity_index
from fris.fris_types import ClassType
from fris.fris_types import DataPoint
from fris.fris_types import DataPointArray
from fris.fris_types import DataPointList
from fris.fris_types import DistanceFunctionType
from fris.fris_types import LabelsArray

ClassificationPoint = tuple[np.ndarray, int]

ClassIndicesDict = dict[ClassType, list[int]]
ClassPointsDict = dict[ClassType, DataPointList]

NEIGHBOURS_INDICES: Final[int] = 1


@dataclass(frozen=True)
class PointNeighbours:
    """Information about point nearest neighbours in point class and from non-point class."""

    # Index of point in the dataset
    point_index: int

    # Index of nearest point from class negation (complement) in the dataset
    nearest_competitor_index: int

    # Index of the nearest other point in the dataset (may belong to same or other class)
    nearest_neighbour_index: int

    # Index of the nearest point of the same class in the dataset
    nearest_inclass_index: Optional[int]

    # Index of the second-nearest point in the dataset
    second_neighbour_index: Optional[int]

    def is_stable(self) -> bool:
        """Indicate, whether point lays inside of class, or at a class boundary.

        This means that nearest neighbour for the point has the same class, as current point.
        """
        return self.nearest_neighbour_index == self.nearest_inclass_index


def compute_class_indices(
    point_indices: Sequence[int], representatives_point_indices: Sequence[int], y: Sequence[ClassType]
) -> tuple[ClassIndicesDict, ClassIndicesDict, list[ClassType]]:
    """Split object indices in `point_indices` using information about classes in `y` into three datastructurs.

    - Dictionary with list of object indices for each class (indices are coming from point_indices)
    - Dictionary with list of object indices for negation of each class (indices here are coming
      from `representative_point_indices`)
    - list of available classes

    Parameters
    ----------
    point_indices - indices of points in the dataset
    representatives_point_indices - indices of the representative points
    y - assignment of classes (for all points in the dataset)
    """
    classes = sorted(list(set(y)))
    class_competitors_indices: ClassIndicesDict = defaultdict(list)
    class_indices: ClassIndicesDict = defaultdict(list)

    representatives_point_indices_set = set(representatives_point_indices)

    for index_x in point_indices:
        c = y[index_x]
        class_indices[c].append(index_x)
        if index_x in representatives_point_indices_set:
            for c_o in classes:
                if c_o != c:
                    class_competitors_indices[c_o].append(index_x)

    return class_indices, class_competitors_indices, classes


def compute_class_points(
    _support_x_indices: Sequence[int], _support_y: Sequence[ClassType], X: np.ndarray
) -> tuple[ClassPointsDict, ClassPointsDict, NDArray[ClassType]]:
    classes = np.array(list(set(_support_y)))
    class_points = {}
    class_competitors_points = {}
    for y_class in classes:
        class_indices = []
        not_class_indices = []
        for i, c in enumerate(_support_y):
            if c == y_class:
                class_indices.append(_support_x_indices[i])
            else:
                not_class_indices.append(_support_x_indices[i])
        class_points[y_class] = [X[i] for i in class_indices]
        class_competitors_points[y_class] = [X[i] for i in not_class_indices]
    return class_points, class_competitors_points, classes


class NearestRepresentativeSearcher:
    """Base interface for finding nearest representative object to class or class negation for a specified point."""

    def __init__(
        self,
        class_representatives: ClassPointsDict,
        class_representatives_competitors: ClassPointsDict,
        distance: DistanceFunctionType,
    ) -> None:
        super().__init__()
        self.class_representatives = class_representatives
        self.class_representatives_competitors = class_representatives_competitors
        self.distance = distance

    def compute_distances_to_nearest_representatives(
        self, point: DataPoint, classes: Sequence[ClassType]
    ) -> np.ndarray:
        raise NotImplementedError()


def nearest_neighbour_from_tree_index(xkd: KDTree, el: np.ndarray, k: int = 1) -> int:
    idx = xkd.query(el, k)[NEIGHBOURS_INDICES]
    if k != 1:
        return cast(int, idx[k - 1])
    return cast(int, idx)


class KDTreeNearestRepresentativeSearcher(NearestRepresentativeSearcher):
    """KDTree based implementation of NearestRepresentativeSearcher."""

    # Currently euclidian distance is implied

    def __init__(
        self,
        class_representatives: ClassPointsDict,
        class_representatives_competitors: ClassPointsDict,
        distance: DistanceFunctionType,
    ) -> None:
        super().__init__(class_representatives, class_representatives_competitors, distance)
        self.representatives_trees = {
            y_class: KDTree(class_representatives[y_class]) for y_class in class_representatives
        }
        self.representative_competitors_trees = {
            y_class: KDTree(class_representatives_competitors[y_class]) for y_class in class_representatives
        }

    @staticmethod
    def nearest_neighbour_from_tree(
        xkd: KDTree, el: DataPoint, elements: Union[np.ndarray, list[np.ndarray]], k: int = 1
    ) -> DataPoint:
        return elements[nearest_neighbour_from_tree_index(xkd, el, k)]

    def find_nearest_representative_for_class(self, point: DataPoint, a_class: ClassType) -> DataPoint:
        return KDTreeNearestRepresentativeSearcher.nearest_neighbour_from_tree(
            self.representatives_trees[a_class], point, self.class_representatives[a_class]
        )

    def find_nearest_representative_for_class_competitors(self, point: DataPoint, a_class: ClassType) -> DataPoint:
        return KDTreeNearestRepresentativeSearcher.nearest_neighbour_from_tree(
            self.representative_competitors_trees[a_class], point, self.class_representatives_competitors[a_class]
        )

    def compute_distances_to_nearest_representatives(
        self, point: DataPoint, classes: Sequence[ClassType]
    ) -> np.ndarray:
        similarities_to_class = np.empty(len(classes))
        for y_class in classes:
            nearest_class_representative_y = self.find_nearest_representative_for_class(point, y_class)
            nearest_competitors_representatives_y = self.find_nearest_representative_for_class_competitors(
                point, y_class
            )

            similarity = rival_similarity(
                point, nearest_class_representative_y, nearest_competitors_representatives_y, self.distance
            )
            similarities_to_class[y_class] = similarity
        return similarities_to_class


class BruteForceNearestRepresentativeSearcher(NearestRepresentativeSearcher):
    """Brute forced based implementation of NearestRepresentativeSearcher."""

    def __init__(
        self,
        class_representatives: ClassPointsDict,
        class_representatives_competitors: ClassPointsDict,
        distance: DistanceFunctionType,
    ) -> None:
        super().__init__(class_representatives, class_representatives_competitors, distance)

    def find_nearest_neighbour(self, point: DataPoint, points_collection: Sequence[DataPoint]) -> Optional[DataPoint]:
        min_distance = float("inf")
        nearest_representative = None
        for p in points_collection:
            point_distance = self.distance(point, p)
            if point_distance < min_distance:
                nearest_representative = p
                min_distance = point_distance
        return nearest_representative

    def find_nearest_representative_for_class(self, point: DataPoint, a_class: ClassType) -> Optional[DataPoint]:
        return self.find_nearest_neighbour(point, self.class_representatives[a_class])

    def find_nearest_representative_for_class_competitors(
        self, point: DataPoint, a_class: ClassType
    ) -> Optional[DataPoint]:
        return self.find_nearest_neighbour(point, self.class_representatives_competitors[a_class])

    def compute_distances_to_nearest_representatives(
        self, point: DataPoint, classes: Sequence[ClassType]
    ) -> np.ndarray:
        min_distances_to_classes = np.empty(len(classes))
        for i, y_class in enumerate(classes):
            min_distance_y = float("inf")
            for y_class_representative in self.class_representatives[y_class]:
                distance_to_y = self.distance(point, y_class_representative)
                if min_distance_y > distance_to_y:
                    min_distance_y = distance_to_y
            min_distances_to_classes[i] = min_distance_y

        # see https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
        partition = np.argpartition(min_distances_to_classes, 1)[:2]

        min_class_index = partition[0]
        next_competitor_index = partition[1]
        similarities_to_class = np.empty(len(classes))
        min_distance_to_representative = min_distances_to_classes[min_class_index]

        for i, y_class in enumerate(classes):
            if i != min_class_index:
                similarities_to_class[i] = fris_function(min_distances_to_classes[i], min_distance_to_representative)
            else:
                assert i == min_class_index
                similarities_to_class[i] = fris_function(
                    min_distance_to_representative, min_distances_to_classes[next_competitor_index]
                )
        return similarities_to_class


def compute_nearest_competitors_indexes(
    class_indices: list[int], neighbours: dict[int, PointNeighbours]
) -> dict[int, int]:
    nearest_competitor = {}
    for class_el_index in class_indices:
        nearest_competitor[class_el_index] = neighbours[class_el_index].nearest_competitor_index
    return nearest_competitor


def compute_nearest_neighbours_among_competitors(
    y: Sequence[ClassType],
    a_class: ClassType,
    class_competitors_indices: list[int],
    classes: list[ClassType],
    neighbours: dict[int, PointNeighbours],
) -> dict[int, int]:
    """Calculate nearest neighbour not belonging to class `a_class` for each element of `class_competitors_indices`."""

    # Map indexes are:
    # Key - index of element in dataset. This element belongs to competitor class
    # Value - index of nearest neighbour among competitors. Index refers to the position of datapoint in the whole
    # dataset
    nearest_neighbour_map_among_competitors = {}
    for comp_index in class_competitors_indices:
        # 2 is used in order to get the neighbour distinct from itself
        # If nearest neighbour from competitor element belongs to currently calculated class,
        # then it cannot be taken, and one need nearest neighbour from another class
        # otherwise, one is safe to take nearest neighbour
        point_neighbours = neighbours[comp_index]
        if point_neighbours.is_stable():
            nearest_neighbour_map_among_competitors[comp_index] = point_neighbours.nearest_neighbour_index
        else:  # nearest_neighbour_to_competitor lays in same class, as a_class
            # if number of classes equals two, then
            # it would be
            if len(classes) == 2:
                if point_neighbours.nearest_inclass_index is not None:
                    nearest_neighbour_map_among_competitors[comp_index] = point_neighbours.nearest_inclass_index
                else:
                    # In this case, the competitor class has only one point
                    assert len(class_competitors_indices) == 1
                    # print("Setting class nearest index among competitors to itself, as class has only one point")
                    nearest_neighbour_map_among_competitors[comp_index] = comp_index
            else:
                if y[point_neighbours.nearest_neighbour_index] != a_class:
                    nearest_neighbour_map_among_competitors[comp_index] = point_neighbours.nearest_neighbour_index
                else:
                    assert y[point_neighbours.nearest_neighbour_index] == a_class
                    assert point_neighbours.second_neighbour_index is not None
                    nearest_neighbour_map_among_competitors[comp_index] = point_neighbours.second_neighbour_index
    return nearest_neighbour_map_among_competitors


def compute_tolerances(
    representative_candidate_index: int,
    class_competitors_indices: list[int],
    nearest_neighbour_within_competitors: dict[int, int],
    f_0: float,
    distance_matrix: np.ndarray,
) -> float:
    tolerances_sum = 0.0
    tolerances_count = 0
    for comp_index in class_competitors_indices:
        tolerance = rival_similarity_index(
            comp_index,
            nearest_neighbour_within_competitors[comp_index],
            representative_candidate_index,
            distance_matrix,
        )
        # Note - in experiments the performance on several datasets is better, if tolerances are
        # not checked to be equal to f_0
        if tolerance >= f_0:
            tolerances_sum += tolerance
            tolerances_count += 1

    if tolerances_count > 0:
        tolerance_of_representative = tolerances_sum / tolerances_count
    else:
        tolerance_of_representative = 0.0
    return tolerance_of_representative


def compute_defensibility(
    representative_candidate_index: int,
    class_indices: list[int],
    nearest_competitors: dict[int, int],
    f_0: float,
    distance_matrix: np.ndarray,
) -> float:
    inner_distances = []
    for cl_el_index in class_indices:
        if cl_el_index == representative_candidate_index:
            continue
        else:
            f_el = rival_similarity_index(
                cl_el_index, representative_candidate_index, nearest_competitors[cl_el_index], distance_matrix
            )
            if f_el >= f_0:
                inner_distances.append(f_el)
    # one is added in order to take into account stolp itself
    stolp_defensibility = (sum(inner_distances) + 1.0) / (len(inner_distances) + 1)
    return stolp_defensibility


def compute_efficiencies(
    class_indices: list[int],
    class_competitors_indices: list[int],
    nearest_competitor: dict[int, int],
    nearest_neighbours_within_competitors: dict[int, int],
    f_0: float,
    _lambda: float,
    distance_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    defensibilities = []
    tolerances = []
    efficiencies = []
    for stolp_candidate_index in class_indices:
        defensibility = compute_defensibility(
            stolp_candidate_index, class_indices, nearest_competitor, f_0, distance_matrix
        )
        # important to take neighbour of element of class b, and not element itself.
        tolerance = compute_tolerances(
            stolp_candidate_index,
            class_competitors_indices,
            nearest_neighbours_within_competitors,
            f_0,
            distance_matrix,
        )

        efficiency = _lambda * defensibility + (1.0 - _lambda) * tolerance

        defensibilities.append(defensibility)
        tolerances.append(tolerance)
        efficiencies.append(efficiency)

    return np.array(defensibilities), np.array(tolerances), np.array(efficiencies)


def compute_support_x_y(
    classes: Sequence[ClassType], class_stolps_indices: dict[ClassType, list[int]]
) -> tuple[list[int], list[ClassType]]:
    support_x_indices = []
    support_y = []
    for y_class in classes:
        y_class_stolp_indices = class_stolps_indices[y_class]
        support_x_indices.extend(y_class_stolp_indices)
        support_y.extend([y_class] * len(y_class_stolp_indices))
    return support_x_indices, support_y


def classify_fris_stolp(
    points: DataPointArray, classes: Sequence[ClassType], nearest_representative_searcher: NearestRepresentativeSearcher
) -> np.ndarray:
    result = []
    for point in points:
        similarities_to_classes = nearest_representative_searcher.compute_distances_to_nearest_representatives(
            point, classes
        )
        class_value = cast(int, np.argmax(similarities_to_classes))
        result.append(classes[class_value])
    end_result = np.array(result)
    return end_result


def has_conflicting_points(X: DataPointArray, y: LabelsArray) -> bool:
    """Check for points in X that have differing labels in y.

    Parameters:
    X (array-like): Array of points (float values or vectors).
    y (array-like): Array of labels corresponding to points in X.

    Returns:
     Whether points with conflicting labels exist
    """
    # Dictionary to store points and their associated labels
    point_label_dict = defaultdict(set)

    # Iterate over all points and labels to fill the dictionary
    for point, label in zip(X, y):
        point_label_dict[tuple(point)].add(label)

    for label_set in point_label_dict.values():
        if len(label_set) > 1:
            return True
    return False


class FrisStolpWithDistanceCaching(BaseEstimator, ClassifierMixin):
    """Implementation of FRiS-Stolp algorithm with a cached matrix for storing pre-computed distances between points."""

    def __init__(
        self,
        distance: DistanceFunctionType = fast_euclidean_distance,
        f_0: float = 0.0,
        _lambda: float = 0.5,
        nearest_reference_searcher_factory: Callable[
            [ClassPointsDict, ClassPointsDict, DistanceFunctionType], NearestRepresentativeSearcher
        ] = BruteForceNearestRepresentativeSearcher,
        do_second_round: bool = True,
        check_for_errors: bool = True,
        allocate_points_to_nearest_stolp: bool = True,
    ) -> None:
        self.distance = distance
        self.f_0 = f_0
        self._lambda = _lambda
        self.nearest_reference_searcher_factory = nearest_reference_searcher_factory
        self.do_second_round = do_second_round
        self.check_for_errors = check_for_errors
        self.allocate_points_to_nearest_stolp = allocate_points_to_nearest_stolp

    def init_distance_matrix(self, points_X: DataPointArray, points_y: Sequence[ClassType]) -> None:
        if len(points_X) != len(points_y):
            raise ValueError("Different shapes for points and classifications")
        # Distance matrix is computed only once
        self.distance_matrix = FrisStolpWithDistanceCaching.compute_distance_matrix_cdist(points_X, self.distance)

    def initialize_neighbours(self, points_y: Sequence[ClassType], points_indices: list[int]) -> None:
        self.neighbours = self.compute_neighbours(points_indices, points_indices, points_y, False)

    def compute_neighbours(
        self,
        points_indices: list[int],
        searched_point_indices: list[int],
        points_y: Sequence[ClassType],
        include_self: bool = False,
    ) -> dict[int, PointNeighbours]:
        neighbours = {}
        num_points = len(points_indices)
        searched_num_points = len(searched_point_indices)
        for i in range(num_points):
            i_index = points_indices[i]
            c_i = points_y[i_index]
            # brute force calculation

            min_distance = float("inf")
            min_distance_same_class = float("inf")
            min_distance_competitor = float("inf")

            nearest_neighbour_index = None

            nearest_neighbour_intraclass_index = None

            nearest_neighbour_competitor_index = None
            second_nearest_neighbour_index = None

            for j in range(searched_num_points):
                j_index = searched_point_indices[j]
                if i_index == j_index and not include_self:
                    continue
                c_j = points_y[j_index]
                distance_i_j = self.distance_matrix[i_index, j_index]
                if distance_i_j < min_distance:
                    min_distance = distance_i_j
                    nearest_neighbour_index = j_index

                if c_i == c_j:
                    if distance_i_j < min_distance_same_class:
                        min_distance_same_class = distance_i_j
                        nearest_neighbour_intraclass_index = j_index
                else:
                    assert c_i != c_j
                    if distance_i_j < min_distance_competitor:
                        min_distance_competitor = distance_i_j
                        nearest_neighbour_competitor_index = j_index

            if nearest_neighbour_index is not None and nearest_neighbour_competitor_index is not None:
                if points_y[i_index] == points_y[nearest_neighbour_index]:
                    # point i_index and nearest neighbour lay in the same class
                    assert nearest_neighbour_index == nearest_neighbour_intraclass_index
                    # point is stable - no need for second-nearest neighbour
                else:
                    # points_y[i_index] != points_y[nearest_neighbour_index]
                    # i_index and nearest_neighbour_index lay in the different class
                    assert nearest_neighbour_index == nearest_neighbour_competitor_index
                    # point is unstable. Need one more point, which has different class from
                    second_min_distance = float("inf")
                    second_nearest_neighbour_index = None
                    nearest_competitor_class = points_y[nearest_neighbour_index]
                    for j in range(num_points):
                        j_index = points_indices[j]
                        if j_index in [i_index, nearest_neighbour_index]:
                            continue
                        if points_y[j_index] == nearest_competitor_class:
                            continue
                        distance_i_j = self.distance_matrix[i_index, j_index]
                        if distance_i_j < second_min_distance:
                            second_min_distance = distance_i_j
                            second_nearest_neighbour_index = j

                neighbours[i_index] = PointNeighbours(
                    i_index,
                    nearest_neighbour_competitor_index,
                    nearest_neighbour_index,
                    nearest_neighbour_intraclass_index,
                    second_nearest_neighbour_index,
                )
            else:
                raise NotImplementedError("To be implemented")
        return neighbours

    @staticmethod
    def compute_distance_matrix_cdist(
        points_X: np.ndarray, distance: Callable[[DataPoint, DataPoint], float]
    ) -> np.ndarray:
        """Compute distance matrix between points from `points_X` using provided `distance` function.

        This is used for speeding up computations.
        """
        return cast(np.ndarray, cdist(points_X, points_X, distance))

    def find_nearest_neighbour(self, X: DataPoint, index: int = 0) -> Optional[DataPoint]:
        neighbour_index = self.neighbours[index].nearest_neighbour_index
        if neighbour_index is not None:
            return cast(DataPoint, X[neighbour_index])
        return None

    def compute_set_rival_similarity(
        self, point_indices: list[int], representative_point_indices: list[int], y: Sequence[ClassType]
    ) -> float:
        class_compactness_map_fris = self.compute_classes_summary_compactness(
            point_indices, representative_point_indices, y
        )
        compactness_for_classes_fris = class_compactness_map_fris.values()
        return average(compactness_for_classes_fris)

    def get_class_compactness_map(self) -> dict[ClassType, float]:
        return self.compute_classes_summary_compactness(list(range(len(self._y))), self.support_x_indices_, self._y)

    def compute_class_efficiency_compactness_map(
        self, point_indices: list[int], y_full: Sequence[ClassType]
    ) -> dict[ClassType, float]:
        # TODO: Reconsider the calculation
        # Currently the efficiencies, obtained during process of finding
        # cover of class with stolps are being used.
        # This means, that:
        # 1. It is to be clarified, which stolp is taken in order to calculate the efficiency
        # of element (i.e., nearest stolp or first covering stolp)
        # 2. In computing defensibility for the element, it gets changed with every round

        class_elements, _, classes = compute_class_indices(point_indices, point_indices, y_full)

        class_compactness_map: dict[ClassType, float] = {}
        for y_class in classes:
            y_class_elements = class_elements[y_class]
            class_efficiency_sum = sum(
                self.element_efficiences[class_elements_index] for class_elements_index in y_class_elements
            )
            class_compactness_map[y_class] = class_efficiency_sum / len(y_class_elements)
        return class_compactness_map

    def compute_dataset_compactness(self, point_indices: list[int], y_full: Sequence[ClassType]) -> float:
        representative_neighbours = self.compute_neighbours(point_indices, point_indices, y_full, include_self=False)
        f_sum = 0.0
        for element_index in point_indices:
            f_el = rival_similarity_index(
                element_index,
                # TODO: consider, whether here one could really get None value
                cast(int, representative_neighbours[element_index].nearest_inclass_index),
                representative_neighbours[element_index].nearest_competitor_index,
                self.distance_matrix,
            )
            f_sum += f_el
        return f_sum / len(point_indices)

    def compute_classes_summary_compactness(
        self,
        point_indices: list[int],
        representative_point_indices: list[int],
        y_full: Sequence[ClassType],
        include_self: bool = True,
    ) -> dict[ClassType, float]:
        representative_neighbours = self.compute_neighbours(
            point_indices, representative_point_indices, y_full, include_self=include_self
        )
        class_elements, _, classes = compute_class_indices(point_indices, point_indices, y_full)

        class_compactness_map: dict[ClassType, float] = {}
        for y_class in classes:
            y_class_element_indices = class_elements[y_class]
            f_class_sum = 0.0
            for class_element_index in y_class_element_indices:
                f_el = rival_similarity_index(
                    class_element_index,
                    # TODO: consider, whether here one could really get None value
                    cast(int, representative_neighbours[class_element_index].nearest_inclass_index),
                    representative_neighbours[class_element_index].nearest_competitor_index,
                    self.distance_matrix,
                )
                f_class_sum += f_el
            class_compactness_map[y_class] = f_class_sum / len(y_class_element_indices)
        return class_compactness_map

    def compute_stolp_clusters(
        self, point_indices: list[int], representative_point_indices: list[int], y_full: Sequence[ClassType]
    ) -> dict[ClassType, list[int]]:
        representative_neighbours = self.compute_neighbours(
            point_indices, representative_point_indices, y_full, include_self=True
        )
        class_elements, _, classes = compute_class_indices(point_indices, point_indices, y_full)

        stolp_elements: dict[ClassType, list[int]] = defaultdict(list)
        for y_class in classes:
            y_class_element_indices = class_elements[y_class]
            for class_element_index in y_class_element_indices:
                # TODO: consider, whether one could get here None Values
                nearest_own_stolp = cast(int, representative_neighbours[class_element_index].nearest_inclass_index)
                nearest_competitor_stolp = representative_neighbours[class_element_index].nearest_competitor_index
                f_el = rival_similarity_index(
                    class_element_index, nearest_own_stolp, nearest_competitor_stolp, self.distance_matrix
                )
                if self.check_for_errors:
                    if f_el >= self.f_0:
                        stolp_elements[nearest_own_stolp].append(class_element_index)
                    else:
                        raise ValueError(
                            f"Element of {class_element_index} of class {y_class} is far away from nearest "
                            f"stolp {nearest_own_stolp} nearest competitor stolp {nearest_competitor_stolp} "
                            f"f_value {f_el}"
                        )
                else:
                    stolp_elements[nearest_own_stolp].append(class_element_index)
        return stolp_elements

    def print_out_clusters(self, header: str) -> None:
        print(header)
        point_count = 0
        for stolp_index, stolp_element_indices, y_class in zip(
            self.support_x_indices_, self.support_x_point_indices_, self.support_y_
        ):
            print(f"class {y_class} stolp: {stolp_index} Elements: {stolp_element_indices}")
            point_count += len(stolp_element_indices)
        print(f"Total point count {point_count}")

    def _check_inputs(self, X: DataPointArray, y: LabelsArray) -> None:
        if len(np.unique(y)) == 1:
            raise ValueError("Not able to build classifier from examples, containing only one class")
        if has_conflicting_points(X, y):
            raise ValueError("Data Points has conflicting points - same point belongs to different classes")

    def fit(
        self, X: DataPointArray, y: Sequence[ClassType], debug_output: bool = False
    ) -> "FrisStolpWithDistanceCaching":
        """Fit model."""
        self._check_inputs(X, y)
        self.init_distance_matrix(X, y)
        self._X = X
        self._y = y

        self.stolp_elements_indices: dict[int, list[int]] = defaultdict(list)
        for i in range(len(X)):
            self.stolp_elements_indices[i].append(i)

        self.element_efficiences = np.zeros(len(y))
        self.element_tolerances = np.zeros(len(y))
        self.element_defensibility = np.zeros(len(y))

        points_indices = [i for i in range(len(X))]
        X_S_indices, Y_S, stolp_element_indices = self.make_pass(points_indices, y, debug_output)
        self.support_x_indices_, self.support_y_ = X_S_indices, Y_S
        self.stolp_elements_indices = stolp_element_indices

        class_stolps_indices: dict[ClassType, list[int]] = defaultdict(list)
        for index, a_class in zip(self.support_x_indices_, self.support_y_):
            class_stolps_indices[a_class].append(index)

        self.compute_clusters_coverage(class_stolps_indices)

        self.compute_nearest_stolps_clusters(class_stolps_indices, points_indices, y)

        self.all_points_neighbours = self.neighbours
        if debug_output:
            print(X_S_indices, Y_S)
        if debug_output:
            for i in X_S_indices:
                print(X[i].tolist(), y[i])

        if self.do_second_round:
            self.support_x_indices_, self.support_y_, second_round_stolp_element_indices = self.make_pass(
                X_S_indices, y, debug_output
            )
            if debug_output:
                print(self.support_x_indices_, self.support_y_)
            # Update of covered element. Is incorrect and will be removed later
            for stolp_index, cluster_covered_elements_indices in second_round_stolp_element_indices.items():
                for ind in cluster_covered_elements_indices:
                    ind_item = ind
                    if ind_item in self.stolp_elements_indices:
                        if ind_item != stolp_index:
                            # Such extension may be unsafe - as points, covered by one stolp would not be
                            # necessarily covered by another stolp
                            self.stolp_elements_indices[stolp_index].extend(self.stolp_elements_indices[ind_item])
                            del self.stolp_elements_indices[ind_item]
                    else:
                        self.stolp_elements_indices[stolp_index].append(ind_item)

        else:
            self.support_x_indices_, self.support_y_ = X_S_indices, Y_S

        class_stolps_indices = defaultdict(list)
        for index, a_class in zip(self.support_x_indices_, self.support_y_):
            class_stolps_indices[a_class].append(index)

        self.compute_clusters_coverage(class_stolps_indices)

        if self.allocate_points_to_nearest_stolp:
            self.compute_nearest_stolps_clusters(class_stolps_indices, points_indices, y)

        self.support_x_ = np.array([X[i] for i in self.support_x_indices_])

        self.support_x_points_ = [
            np.array([X[i] for i in self.support_x_point_indices_[stolp_index]])
            for stolp_index in range(len(self.support_x_indices_))
        ]

        self._score = self.compute_set_rival_similarity(points_indices, self.support_x_indices_, y)

        self.class_references_, self.class_negation_references_, self.classes_ = compute_class_points(
            self.support_x_indices_, self.support_y_, X
        )
        return self

    def compute_nearest_stolps_clusters(
        self, class_stolps_indices: dict[ClassType, list[int]], points_indices: list[int], y: Sequence[ClassType]
    ) -> None:
        self.support_x_point_indices_: list[list[int]] = []

        elements_stolps_distribution = self.compute_stolp_clusters(points_indices, self.support_x_indices_, y)
        for y_class in class_stolps_indices.keys():
            y_class_stolp_indices = class_stolps_indices[y_class]
            for stolp_index in y_class_stolp_indices:
                self.support_x_point_indices_.append(elements_stolps_distribution[stolp_index])

    def compute_clusters_coverage(self, class_stolps_indices: dict[ClassType, list[int]]) -> None:
        self.support_x_point_indices_ = []
        for y_class in class_stolps_indices.keys():
            y_class_stolp_indices = class_stolps_indices[y_class]
            for stolp_index in y_class_stolp_indices:
                self.support_x_point_indices_.append(self.stolp_elements_indices[stolp_index])

    def get_stolp_index_for_point_index(self, point_index: int) -> int:
        """Return index of representative point for the point with index `point_index`."""
        for i, point_indices in enumerate(self.support_x_point_indices_):
            if point_index in point_indices:
                return self.support_x_indices_[i]
        print(f"{point_index=} {self.support_x_point_indices_=}")
        raise ValueError(f"No stolp found for point index {point_index}")

    def get_point_cluster_size(self, row: int) -> int:
        stolp_index = self.get_stolp_index_for_point_index(row)
        for index, stolp_element_index in enumerate(self.support_x_indices_):
            if stolp_element_index == stolp_index:
                return len(self.support_x_point_indices_[index])
        raise ValueError(f"No cluster found for point with index {row}")

    def make_pass(
        self, point_indices: list[int], y_full: Sequence[ClassType], debug_output: bool = False
    ) -> tuple[list[int], list[ClassType], dict[int, list[int]]]:
        neighbours = self.compute_neighbours(point_indices, point_indices, y_full, include_self=False)
        self.neighbours = neighbours

        class_elements, class_competitors_elements, classes = compute_class_indices(
            point_indices, point_indices, y_full
        )
        return self.compute_coverage_by_stolps(
            classes, class_elements, class_competitors_elements, debug_output, neighbours, y_full
        )

    def compute_coverage_by_stolps(
        self,
        classes: list[ClassType],
        class_elements: dict[ClassType, list[int]],
        class_competitors_elements: dict[ClassType, list[int]],
        debug_output: bool,
        neighbours: dict[int, PointNeighbours],
        y_full: Sequence[ClassType],
    ) -> tuple[list[int], list[ClassType], dict[int, list[int]]]:

        class_stolps_indices: dict[ClassType, list[int]] = defaultdict(list)
        stolp_covered_elements_indices: dict[int, list[int]] = {}
        for a_class in classes:
            # doing steps 1-5
            class_indices = class_elements[a_class]
            class_competitors_indices = class_competitors_elements[a_class]

            nearest_neighbours_within_competitors: dict[int, int] = compute_nearest_neighbours_among_competitors(
                y_full, a_class, class_competitors_indices, classes, neighbours
            )

            nearest_competitor: dict[int, int] = compute_nearest_competitors_indexes(class_indices, neighbours)

            prev_len = len(class_indices) + 1

            while len(class_indices) > 0 and len(class_indices) != prev_len:
                # should be currently recomputed due to dependency of key on position of element
                # between class objects. If key was run independent, this recomputation could be avoided

                # print("Start of cycle", len(class_indices), class_indices, type(class_indices))

                defensibilities, tolerances, efficiencies = compute_efficiencies(
                    class_indices,
                    class_competitors_indices,
                    nearest_competitor,
                    nearest_neighbours_within_competitors,
                    self.f_0,
                    self._lambda,
                    self.distance_matrix,
                )

                # 3. Then we find object ai with the maximum value
                # of Fi and set it to be the first stolp A11 of the first cluster
                # C11 of the first pattern S1.

                for i, defensibility in zip(class_indices, defensibilities):
                    self.element_defensibility[i] = defensibility
                for i, tol in zip(class_indices, tolerances):
                    self.element_tolerances[i] = tol
                for i, eff in zip(class_indices, efficiencies):
                    self.element_efficiences[i] = eff

                max_index = efficiencies.argmax()
                stolp_index = class_indices[max_index]
                max_efficiency = efficiencies[max_index]
                if debug_output:
                    print("find_representatives", a_class, "stolp_index", stolp_index, "max_efficiency", max_efficiency)

                # 4. We eliminate mi objects that enter in the first cluster from the first pattern.
                # Then, repeating steps 1-3 for
                # other objects of the first pattern, we find the next stolp.
                # The process is terminated as soon as all objects of the
                # first pattern are included in their clusters.

                cluster_covered_elements_indices = [
                    index
                    for index in class_indices
                    if rival_similarity_index(index, stolp_index, nearest_competitor[index], self.distance_matrix)
                    >= self.f_0
                ]

                to_be_deleted = cluster_covered_elements_indices
                class_stolps_indices[a_class].append(stolp_index)

                stolp_covered_elements_indices[stolp_index] = cluster_covered_elements_indices[:]

                prev_len = len(class_indices)

                new_class_indices = list(set(class_indices) - set(to_be_deleted))
                if debug_output:
                    print(
                        "After delete number of new class indices",
                        len(new_class_indices),
                        "removed count",
                        len(to_be_deleted),
                        "Left indices in class",
                        new_class_indices,
                    )
                    print("Removed indices:", to_be_deleted)
                class_indices = new_class_indices

                # 5. We reestablish all objects of pattern S1 and repeat
                # steps 1-4 for all other patterns.

                # 6. The stolps were chosen under the conditions that
                # they were opposed by all objects of the rival patterns.
                # Now patterns are represented only by their stolps. To
                # specify the content of the clusters, we recognize the
                # belonging of all objects to clusters in the conditions
                # where function F is determined by the distances to the
                # nearest friend stolp and the nearest foe stolp. The content of the clusters can be changed.
        # 7. Finally, we have to find the average value Fs of
        # the similarity functions for all objects and their stolps.
        # The quantity Fs characterizes the quality of the system
        # training and is closely connected with the errors that
        # occur during the recognition of the control objects.
        # The output of the FRiS-Stolp algorithm is the decision rule that consists of a list of standards (stolps) that
        # describe each pattern, a list of objects belonging to each
        # cluster, the values of internal distances for each cluster,
        # and the average value Fs of the similarity functions.
        support_x_indices, support_y = compute_support_x_y(classes, class_stolps_indices)
        return support_x_indices, support_y, stolp_covered_elements_indices

    def predict(self, X: DataPointArray) -> np.ndarray:
        """Predict the entries of."""
        return classify_fris_stolp(X, self.classes_, self.create_nearest_representative_searcher())

    def create_nearest_representative_searcher(self) -> NearestRepresentativeSearcher:
        return self.nearest_reference_searcher_factory(
            self.class_references_, self.class_negation_references_, self.distance
        )

    def decision_function(self, X: DataPointArray) -> np.ndarray:
        """Return for the set of points a value of fris function for every class."""
        nearest_representative_searcher = self.create_nearest_representative_searcher()
        classes = self.classes_
        result = []
        for point in X:
            similarities_to_classes = nearest_representative_searcher.compute_distances_to_nearest_representatives(
                point, classes
            )
            result.append(similarities_to_classes)
        return np.array(result)

    def predict_proba(self, X: DataPointArray) -> np.ndarray:
        # For two class case: find 2 representatives from nearest classes
        # compute fris function on them f= fris(x, nearest_stolp(class_1), nearest_stolp(class_2))
        # probability belonging to class approximates as ((f+1)/2)
        similarities_to_classes = self.decision_function(X)
        normalized_probabilities = (similarities_to_classes + 1.0) / 2.0
        normalized_probabilities /= np.sum(normalized_probabilities, axis=1)[:, np.newaxis]
        return normalized_probabilities


class FrisStolpWithDistanceCachingCorrected(FrisStolpWithDistanceCaching):
    """Another version of FrisStolpWithDistanceCaching implementation."""

    def __init__(
        self,
        distance: DistanceFunctionType = fast_euclidean_distance,
        f_0: float = 0.0,
        _lambda: float = 0.5,
        nearest_reference_searcher_factory: Callable[
            [ClassPointsDict, ClassPointsDict, DistanceFunctionType], NearestRepresentativeSearcher
        ] = BruteForceNearestRepresentativeSearcher,
        do_second_round: bool = True,
        check_for_errors: bool = True,
        allocate_points_to_nearest_stolp: bool = True,
    ) -> None:
        super().__init__(
            distance,
            f_0,
            _lambda,
            nearest_reference_searcher_factory,
            do_second_round,
            check_for_errors,
            allocate_points_to_nearest_stolp,
        )

    def redistribute_points_to_clusters(
        self,
        representative_point_indices: list[int],
        x_full: DataPointArray,
        y_full: Sequence[ClassType],
        debug_output: bool = False,
    ) -> tuple[list[int], list[ClassType], dict[int, list[int]]]:
        point_indices = list(range(len(x_full)))
        neighbours = self.compute_neighbours(point_indices, representative_point_indices, y_full, include_self=True)

        class_elements, class_competitors_elements, classes = compute_class_indices(
            point_indices, representative_point_indices, y_full
        )
        return self.compute_coverage_by_stolps(
            classes, class_elements, class_competitors_elements, debug_output, neighbours, y_full
        )

    def fit(
        self, X: DataPointArray, y: Sequence[ClassType], debug_output: bool = False
    ) -> "FrisStolpWithDistanceCaching":
        self._check_inputs(X, y)
        self.init_distance_matrix(X, y)
        self._X = X
        self._y = y

        # dictionary, which saves for each stolp indices of elements, which belong to this stolp
        self.stolp_elements_indices: dict[int, list[int]] = defaultdict(list)
        for i in range(len(X)):
            self.stolp_elements_indices[i].append(i)

        self.element_efficiences = np.zeros(len(y))
        self.element_tolerances = np.zeros(len(y))
        self.element_defensibility = np.zeros(len(y))

        points_indices = list(range(len(X)))
        self.support_x_indices_, self.support_y_, stolp_element_indices = self.make_pass(
            points_indices, y, debug_output
        )
        self.all_points_neighbours = self.neighbours
        self.stolp_elements_indices = stolp_element_indices

        if debug_output:
            for i in self.support_x_indices_:
                print(X[i].tolist(), y[i])

        if self.do_second_round:
            self.support_x_indices_, self.support_y_, stolp_element_indices = self.redistribute_points_to_clusters(
                self.support_x_indices_, X, y, debug_output
            )
            self.stolp_elements_indices = stolp_element_indices

        class_stolps_indices: dict[ClassType, list[int]] = defaultdict(list)
        for index, a_class in zip(self.support_x_indices_, self.support_y_):
            class_stolps_indices[a_class].append(index)

        self.compute_clusters_coverage(class_stolps_indices)

        if self.allocate_points_to_nearest_stolp:
            self.compute_nearest_stolps_clusters(class_stolps_indices, points_indices, y)

        self.support_x_ = np.array([X[i] for i in self.support_x_indices_])

        self.support_x_points_ = [
            np.array([X[i] for i in self.support_x_point_indices_[stolp_index]])
            for stolp_index in range(len(self.support_x_indices_))
        ]

        self._score = self.compute_set_rival_similarity(points_indices, self.support_x_indices_, y)

        self.class_references_, self.class_negation_references_, self.classes_ = compute_class_points(
            self.support_x_indices_, self.support_y_, X
        )
        return self
