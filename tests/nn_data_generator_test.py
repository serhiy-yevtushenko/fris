# type:ignore
import numpy as np

from fris.nn_data_generator import generate_separable_data
from fris.nn_data_generator import generate_separable_data_with_boundary
from fris.nn_data_generator import saw_point
from fris.nn_data_generator import stripe_number


def test_generate_separable_data():
    X, y = generate_separable_data(200, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
    assert X.shape == (200, 2)
    assert y.shape == (200,)
    assert np.sum(y == -1) == 100
    assert np.sum(y == 1) == 100
    assert np.all(X[y == -1][:, 1] <= 0.3)
    assert np.all(X[y == 1][:, 1] >= 0.7)


def test_generate_separable_data_with_boundary():
    X, y = generate_separable_data_with_boundary(200, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
    assert X.shape == (200, 2)
    assert y.shape == (200,)
    assert np.sum(y == -1) == 100
    assert np.sum(y == 1) == 100
    assert np.all(X[y == -1][:, 1] <= 0.5)
    assert np.all(X[y == 1][:, 1] >= 0.5)


def test_stripe_number():
    assert stripe_number(0.0, 0.1) == 0
    assert stripe_number(0.09, 0.1) == 0
    assert stripe_number(0.1, 0.1) == 1
    assert stripe_number(0.9, 0.1) == 9
    assert stripe_number(0.99, 0.1) == 9
    assert stripe_number(1.0, 0.1) == 10


def test_saw_point():
    assert saw_point(0, 0)
    assert saw_point(0.1, 0.1)

    assert saw_point(0, 0.1)
    assert saw_point(0.1, 0.89)
    assert not saw_point(0.1, 0.91)
    assert saw_point(0.2, 0.1)
    assert not saw_point(0, 1.0)
    assert not saw_point(0, 0.91)
