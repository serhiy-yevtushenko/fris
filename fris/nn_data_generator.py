"""Utilities for generating test data with different kind of difficulty for Nearest-Neighbours based classifiers."""

from collections.abc import Callable
from collections.abc import Sequence
from functools import partial
from typing import Final

import numpy as np
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from fris.fris_types import DataPointArray
from fris.fris_types import LabelsArray

FIRST_CLASS_LABEL: Final[int] = -1
SECOND_CLASS_LABEL: Final[int] = 1


def generate_separable_two_class_data(
    n_points: int,
    first_class_upper_boundary: float,
    second_class_low_boundary: float,
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    first_class_point_count = n_points // 2
    first_class_y = y_min + np.random.uniform(0, first_class_upper_boundary, first_class_point_count) * (y_max - y_min)
    first_class_x = x_min + np.random.uniform(0, 1, first_class_point_count) * (x_max - x_min)
    first_class_points = np.array([(x, y) for x, y in zip(first_class_x, first_class_y)])
    first_class_labels = np.full((first_class_point_count,), FIRST_CLASS_LABEL)

    second_class_point_count = n_points - first_class_point_count
    second_class_y = y_min + np.random.uniform(second_class_low_boundary, 1, second_class_point_count) * (y_max - y_min)
    second_class_x = x_min + np.random.uniform(0, 1, second_class_point_count) * (x_max - x_min)
    second_class_points = np.array([(x, y) for x, y in zip(second_class_x, second_class_y)])

    second_class_labels = np.full((second_class_point_count,), SECOND_CLASS_LABEL)

    return (
        np.concatenate((first_class_points, second_class_points), axis=0),
        np.concatenate((first_class_labels, second_class_labels), axis=0),
    )


def generate_separable_data(
    n_points: int, x_min: float = -50.0, x_max: float = 50.0, y_min: float = -50.0, y_max: float = 50.0
) -> tuple[np.ndarray, np.ndarray]:
    return generate_separable_two_class_data(n_points, 0.3, 0.7, x_min, x_max, y_min, y_max)


def generate_separable_data_with_boundary(
    n_points: int, x_min: float = -50.0, x_max: float = 50.0, y_min: float = -50.0, y_max: float = 50.0
) -> tuple[np.ndarray, np.ndarray]:
    return generate_separable_two_class_data(n_points, 0.5, 0.5, x_min, x_max, y_min, y_max)


def stripe_number(x: float, stripe_width: float) -> int:
    assert 0.0 <= x <= 1.0, f"{x} is not between 0 and 1"
    assert 0.0 < stripe_width <= 1.0
    stripe_id = int(x / stripe_width)
    return stripe_id


def generate_stripes_data(
    n_points: int,
    n_stripes: int = 10,
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    stripe_width = 1.0 / n_stripes
    all_class_labels = []

    x_pure = np.random.uniform(0, 1, n_points)
    a_class_x = x_min + x_pure * (x_max - x_min)
    a_class_y = y_min + np.random.uniform(0, 1, n_points) * (y_max - y_min)
    a_class_points = np.array([(x, y) for x, y in zip(a_class_x, a_class_y)])
    for x_pure_p in x_pure:
        all_class_labels.append(FIRST_CLASS_LABEL if stripe_number(x_pure_p, stripe_width) % 2 else SECOND_CLASS_LABEL)

    return (a_class_points, np.array(all_class_labels))


def y_below(_x: float, y: float, y_value: float = 0.5) -> bool:
    return y < y_value


def wide_saw_point(x: float, y: float) -> bool:
    return saw_point(x, y, 0.1, 0.9)


def narrow_saw_point(x: float, y: float) -> bool:
    return saw_point(x, y, -0.4, 1.4)


def islands_point(x: float, y: float, stripe_width: float = 0.1) -> bool:
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0

    x_parts = int(x / stripe_width)
    y_parts = int(y / stripe_width)
    return x_parts % 2 != 0 and y_parts % 2 != 0


def checkers_point(x: float, y: float, stripe_width: float = 0.1) -> bool:
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0

    x_parts = int(x / stripe_width)
    y_parts = int(y / stripe_width)
    return x_parts % 2 != y_parts % 2


def stripes_point(x: float, _y: float, stripe_width: float = 0.1) -> bool:
    return stripe_number(x, stripe_width) % 2 != 0


def saw_point(x: float, y: float, low_y: float = 0.1, upper_y: float = 0.9, stripe_width: float = 0.1) -> bool:
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0

    d_y = int((upper_y - low_y) / stripe_width)

    x_parts = int(x / stripe_width)

    down_direction = x_parts % 2

    if down_direction:
        y_line = upper_y - d_y * (x - stripe_width * x_parts)
    else:
        y_line = low_y + d_y * (x - stripe_width * x_parts)
    return y <= y_line


def predict(x_p: float, y_p: float, classifier: Callable[[float, float], bool]) -> int:
    return FIRST_CLASS_LABEL if classifier(x_p, y_p) else SECOND_CLASS_LABEL


PredictorInputType = tuple[Sequence[float], Sequence[float]]


def predictor(points: PredictorInputType, classifier: Callable[[float, float], bool]) -> np.ndarray:
    x, y = points
    results = []
    for x_p, y_p in zip(x, y):
        results.append(predict(x_p, y_p, classifier))
    return np.array(results)


def narrow_saw_predictor(points: PredictorInputType) -> np.ndarray:
    return predictor(points, narrow_saw_point)


def wide_saw_predictor(points: PredictorInputType) -> np.ndarray:
    return predictor(points, wide_saw_point)


def checker_predictor(points: PredictorInputType, stripe_width: float = 1.0 / 20) -> np.ndarray:
    return predictor(points, partial(checkers_point, stripe_width=stripe_width))


def stripes_predictor(points: PredictorInputType, stripe_width: float = 1.0 / 20) -> np.ndarray:
    return predictor(points, partial(stripes_point, stripe_width=stripe_width))


def y_below_predictor(points: PredictorInputType, y_value: float = 0.5) -> np.ndarray:
    return predictor(points, partial(y_below, y_value=y_value))


def islands_predictor(points: PredictorInputType, y_value: float = 0.5) -> np.ndarray:
    del y_value  # unused
    return predictor(points, islands_point)


def generate_classifier_data(
    n_points: int,
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
    predictor: Callable[[tuple[Sequence[float], Sequence[float]]], np.ndarray] = wide_saw_predictor,
) -> tuple[np.ndarray, np.ndarray]:
    all_class_labels = []

    x_pure = np.random.uniform(0, 1, n_points)
    y_pure = np.random.uniform(0, 1, n_points)

    a_class_x = x_min + x_pure * (x_max - x_min)
    a_class_y = y_min + y_pure * (y_max - y_min)

    a_class_points = np.array([(x, y) for x, y in zip(a_class_x, a_class_y)])

    for x_p, y_p in zip(x_pure, y_pure):
        all_class_labels.append(predictor(([x_p], [y_p])))
    return (a_class_points, np.array(all_class_labels).squeeze())


def _rescale_to_50_50_area(X: DataPointArray) -> DataPointArray:
    X = StandardScaler().fit_transform(X)
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    diameter = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
    X = X / diameter * 50
    return X


def generate_circles_points(point_count: int) -> tuple[DataPointArray, LabelsArray]:
    X, y = make_circles(n_samples=point_count, noise=0.2, factor=0.5, random_state=1)
    y[y == 0] = FIRST_CLASS_LABEL
    return _rescale_to_50_50_area(X), y


def generate_moon_points(point_count: int) -> tuple[DataPointArray, LabelsArray]:
    X, y = make_moons(n_samples=point_count, noise=0.2, random_state=1)
    y[y == 0] = FIRST_CLASS_LABEL
    return _rescale_to_50_50_area(X), y
