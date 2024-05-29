# type:ignore
from itertools import combinations
from typing import List
from typing import Tuple

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

from fris.nn_utils import binomial_coefficient
from fris.nn_utils import compactness_profile
from fris.nn_utils import complete_cross_validation
from fris.nn_utils import compute_ccvs
from fris.nn_utils import gamma_ccv


def n_out_of_x_split(X, n: int) -> List[Tuple[List[int], List[int]]]:
    point_index = [i for i in range(len(X))]
    result = []
    for train_index in combinations(point_index, n):
        train_index_indices = set(train_index)
        test_index = [i for i in point_index if i not in set(train_index_indices)]
        result.append((list(train_index), test_index))
    return result


def test_n_out_of_x_split():
    X = ["a", "b", "c"]
    result = n_out_of_x_split(X, 2)
    assert len(result) == 3
    train_indices = [r[0] for r in result]
    assert [0, 1] in train_indices
    assert [0, 2] in train_indices
    assert [1, 2] in train_indices


def test_compactness_profile():
    X = [[0], [1], [3], [7]]
    y = [0, 0, 1, 1]

    # pr[1] - NN([0]->0) = ([1]->0, [3]->1, [7]->1),
    #         NN([1]->0) = ([0]->0, [3]->1, [7]->1)
    #         NN([3]->1) = ([1]->0, [0]->0, [7]->1)
    #         NN([7]->1) = ([3]->1, [1]->0, [0]->0)
    #                       1/4          1,     3/4

    profile = compactness_profile(X, y)
    print(profile)

    assert profile == [0.25, 1.0, 0.75]


def generate_normal(x, y, items_num):
    x = [np.random.normal() * 3 + x for i in range(items_num)]  # scale random distribution by 3
    y = [np.random.normal() * 3 + y for i in range(items_num)]
    return x, y


def generate_dataset():
    a = generate_normal(0, 0, 200)
    b = generate_normal(8, 8, 200)
    y1 = [1] * len(a[0])
    y2 = [-1] * len(b[0])
    x = np.concatenate((np.array(a).transpose(), np.array(b).transpose()), axis=0)
    y = np.squeeze(np.array([y1 + y2]).transpose())
    return x, y


def test_compactness_profile_iris():
    X, y = load_iris(return_X_y=True)
    profile = compactness_profile(X, y)
    print(len(profile))
    assert len(profile) == len(X) - 1
    print(profile)


def test_binomial_coefficient():
    assert binomial_coefficient(3, 1) == 3.0
    assert binomial_coefficient(5, 2) == 10.0


def gamma_ccv_definition(m: int, L: int, j: int) -> float:
    """Function gamma for computing CCV (complete cross validation value) with help of compactness profile. This
    function performs computation strictly according to the definition and is used for comparing against optimized
    implementations.

    :param m - size of the training set
    :param L - size of the whole dataset
    :param j - step for the CCV
    """
    return binomial_coefficient(L - 1 - j, m - 1) / binomial_coefficient(L - 1, m)


def test_gamma_ccv():
    assert gamma_ccv_definition(9, 10, 1) == 1.0
    assert gamma_ccv(9, 10, 1) == gamma_ccv_definition(9, 10, 1)
    assert gamma_ccv(8, 10, 1) == gamma_ccv_definition(8, 10, 1)

    assert gamma_ccv_definition(3, 5, 1) == 0.75
    assert gamma_ccv(3, 5, 1) == 0.75
    assert gamma_ccv_definition(3, 5, 2) == 0.25
    assert gamma_ccv(3, 5, 2) == 0.25


def check_leave_one_out(x: List[float], y: List[float]):
    X = np.array(x).reshape((-1, 1))
    y = np.array(y).reshape((-1,))
    cp = compactness_profile(X, y)
    print(cp)

    loo = LeaveOneOut()
    acc = 0
    for train_index, test_index in loo.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        one_NN = KNeighborsClassifier(n_neighbors=1)
        one_NN.fit(x_train, y_train)
        y_pred_test = one_NN.predict(x_test)
        # print(test_index, y_test, y_pred_test)
        acc += (y_test == y_pred_test).all()
    # print(acc)

    ccv_loo = complete_cross_validation(X, y, 1)
    assert ccv_loo == pytest.approx(1.0 - (acc / len(X)))


def check_leave_k_out(x: List[float], y: List[float], k: int):
    X = np.array(x).reshape((-1, 1))
    y = np.array(y).reshape((-1,))
    cp = compactness_profile(X, y)
    print(f"{cp=}")

    correct = 0
    count = 0
    dataset_len = len(X)
    for train_index, test_index in n_out_of_x_split(X, dataset_len - k):
        print(train_index, test_index)
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        one_NN = KNeighborsClassifier(n_neighbors=1)
        one_NN.fit(x_train, y_train)
        y_pred_test = one_NN.predict(x_test)
        correctly_classified = (y_test == y_pred_test).sum()
        correct += correctly_classified
        count += 1

        print(test_index, y_test, y_pred_test, correctly_classified, correct, k * count)

    print(f"{correct=}")

    ccv_lko = complete_cross_validation(X, y, k)
    print(f"{ccv_lko=} {correct=} {k=} {count=} {(1.0 - (correct / (k * count)))=}")
    assert ccv_lko == pytest.approx(1.0 - (correct / (k * count)))


def test_complete_cross_validation_compact():
    check_leave_one_out([0, 1, 2, 6, 7], [-1, -1, -1, 1, 1])
    check_leave_k_out([0, 1, 2, 6, 7], [-1, -1, -1, 1, 1], k=1)
    check_leave_k_out([0, 1, 2, 6, 7], [-1, -1, -1, 1, 1], k=2)
    check_leave_k_out([0, 1, 2, 6, 7], [-1, -1, -1, 1, 1], k=3)
    check_leave_k_out([0, 1, 2, 6, 7], [-1, -1, -1, 1, 1], k=4)


def test_complete_cross_validation_not_compact():
    check_leave_one_out([0, 1, 2, 3, 4], [-1, 1, -1, 1, -1])
    check_leave_k_out([0, 1, 2, 3, 4], [-1, 1, -1, 1, -1], k=1)
    check_leave_k_out([0, 1, 2, 3, 4], [-1, 1, -1, 1, -1], k=2)
    check_leave_k_out([0, 1, 2, 3, 4], [-1, 1, -1, 1, -1], k=3)
    check_leave_k_out([0, 1, 2, 3, 4], [-1, 1, -1, 1, -1], k=4)


def test_compute_ccvs():
    X = np.array([0, 1, 2, 6, 7]).reshape((-1, 1))
    y = np.array([-1, -1, -1, 1, 1]).reshape((-1,))
    profile = compactness_profile(X, y)
    assert len(profile) == 4
    # 1, 2, 3, 4
    res = compute_ccvs(profile, 4)
    assert len(res) == 4


def test_compute_ccvs_2():
    X = np.array([0, 1, 2, 6, 7, 8]).reshape((-1, 1))
    y = np.array([-1, -1, -1, 1, 1, 1]).reshape((-1,))
    profile = compactness_profile(X, y)
    assert len(profile) == 5
    # 1, 2, 3, 4
    res = compute_ccvs(profile, 3)
    assert len(res) == 3
    assert 1 in res
    assert 3 in res
    assert 5 in res


def test_compute_ccvs_3():
    X = np.array([0, 1, 2, 6, 7, 8]).reshape((-1, 1))
    y = np.array([-1, -1, -1, 1, 1, 1]).reshape((-1,))
    profile = compactness_profile(X, y)
    assert len(profile) == 5
    # 1, 2, 3, 4
    res = compute_ccvs(profile, 2)
    assert len(res) == 2
    assert 1 in res
    assert 5 in res


def test_compute_ccvs_one_step():
    profile = [1.0]
    res = compute_ccvs(profile, 1)
    assert res == {1: 1.0}
