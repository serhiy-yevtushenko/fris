# type:ignore
from functools import wraps
from time import time

import numpy as np
import pytest
from pytest import approx
from scipy.spatial import KDTree
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from fris.fris_datastructures import BruteForceNearestRepresentativeSearcher
from fris.fris_datastructures import FrisStolpWithDistanceCaching
from fris.fris_datastructures import FrisStolpWithDistanceCachingCorrected
from fris.fris_datastructures import KDTreeNearestRepresentativeSearcher
from fris.fris_functions import fast_euclidean_distance
from fris.fris_functions import fris_function
from fris.fris_functions import geometric_average
from fris.fris_functions import rival_similarity


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap


# TODO: Add hypothesis based test for rival similarity (as well handle bad inputs)


def test_rival_similarity():
    assert approx(rival_similarity(1, 2, 3, fast_euclidean_distance)) == 0.33333333333333331
    assert approx(rival_similarity(1, 1, 3, fast_euclidean_distance)) == 1.0
    assert approx(rival_similarity(3, 1, 3, fast_euclidean_distance)) == -1.0
    assert approx(rival_similarity(2, 1, 3, fast_euclidean_distance)) == 0.0


def test_rival_similarity_examples_zagoruiko():
    # Test examples for the fris function from article
    # "Measure of Similarity and Compactness in  Competitive Space"
    # by Nikolay Zagoruiko.
    # Let we have a triangle with sides
    # (a, b) = 7
    # (a, c) = 3
    # (b, c) = 9
    # F(ca|b)
    assert fris_function(r_1=3, r_2=9) == 0.5
    # F(ab|c)
    assert fris_function(r_1=7, r_2=3) == -0.4
    # F(bc|a)
    assert fris_function(r_1=9, r_2=7) == -0.125
    # F(cb|a)
    assert fris_function(r_1=9, r_2=3) == -0.5
    # F(ba|c)
    assert fris_function(r_1=7, r_2=9) == 0.125
    # F(ac|b)
    assert fris_function(r_1=3, r_2=7) == 0.4


def test_nearest_neighbour_from_tree():
    points = np.array([[0, 1], [1, 1], [1, 1.5]])

    xkd = KDTree(points)
    assert np.array_equal(
        np.array([1, 1]), KDTreeNearestRepresentativeSearcher.nearest_neighbour_from_tree(xkd, np.array([1, 1]), points)
    )
    neigbour = KDTreeNearestRepresentativeSearcher.nearest_neighbour_from_tree(xkd, np.array([1, 1]), points, 2)
    assert np.array_equal(np.array([1, 1.5]), neigbour)


def make_test_dataset():
    first_class_points = [(0, 1), (1, 2), (2, 3)]
    second_class_points = [(2, 4), (4, 5), (5, 6)]
    third_class_points = [(6, 7), (7, 8), (8, 9)]
    return make_three_class_dataset(first_class_points, second_class_points, third_class_points)


def make_one_point_per_class_test_dataset():
    first_class_points = [(0, 1)]
    second_class_points = [(2, 4)]
    third_class_points = [(6, 7)]
    return make_three_class_dataset(first_class_points, second_class_points, third_class_points)


def make_three_class_dataset(first_class_points, second_class_points, third_class_points):
    points = (
        [(p, 0) for p in first_class_points]
        + [(p, 1) for p in second_class_points]
        + [(p, 2) for p in third_class_points]
    )
    print(points)
    points_X = np.array([np.array(p[0]) for p in points])
    points_y = np.array([np.array(p[1]) for p in points])
    return points_X, points_y


def test_finding_neighbours():
    points_X, points_y = make_test_dataset()
    classifier = FrisStolpWithDistanceCaching()
    points_indices = list(range(len(points_X)))

    classifier.init_distance_matrix(points_X, points_y)
    classifier.initialize_neighbours(points_y, points_indices)
    print("Distances", classifier.distance_matrix)
    print("Neighbours", classifier.neighbours)
    assert np.array_equal(np.array([1, 2]), classifier.find_nearest_neighbour(points_X, 0))


# TODO: create bad input tests for rival similarity datasets
# Empty dataset
# Dataset with one class
# DAtaset with same points belonging to different classes
# TODO: Create sklearn compatibility tests
# Dataset with string labels


def test_fit():
    points_X, points_y = make_test_dataset()
    classifier = FrisStolpWithDistanceCaching()
    classifier.fit(points_X, points_y, True)


def test_fit_one_point_per_class_dataset():
    points_X, points_y = make_one_point_per_class_test_dataset()
    classifier = FrisStolpWithDistanceCaching()
    classifier.fit(points_X, points_y, True)


def test_dataset_handles_cross_validation_olivetti():
    random_state = 0
    dataset = fetch_olivetti_faces(shuffle=True, random_state=random_state)
    X = dataset.data
    y = dataset.target
    clf = FrisStolpWithDistanceCaching(check_for_errors=False)
    scorer = make_scorer(f1_score, average="macro")
    cross_val_score(clf, X, y, scoring=scorer, cv=3)


def test_dataset_handles_cross_validation_iris():
    X, y = load_iris(return_X_y=True)
    clf = FrisStolpWithDistanceCaching()
    scorer = make_scorer(f1_score, average="macro")
    cross_val_score(clf, X, y, scoring=scorer, cv=3)


def test_dataset_handles_cross_validation_test_dataset():
    X, y = make_test_dataset()
    clf = FrisStolpWithDistanceCaching()
    scorer = make_scorer(f1_score, average="macro")
    # we are interested, that cross validation passes, and not in specific results
    cross_val_score(clf, X, y, scoring=scorer, cv=3)


def test_classes_is_np_array():
    X, y = make_test_dataset()
    clf = FrisStolpWithDistanceCaching()
    clf.fit(X, y)
    assert isinstance(clf.classes_, np.ndarray)


@timing
def run_on_iris(clf):
    print(f"Iris Classifier:{clf}")
    X, y = load_iris(return_X_y=True)
    print(type(X), type(y))
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Test performance")
    print(classification_report(y_test, y_pred))
    print(len(clf.support_x_), len(clf.support_y_))
    print(clf.support_x_, clf.support_y_)


@timing
def run_on_olivetti(clf, scale: bool, expected_accuracy):
    random_state = 0
    dataset = fetch_olivetti_faces(shuffle=True, random_state=random_state)
    data = dataset.data
    target = dataset.target
    # print(data.shape)
    X = data
    # print("X shape:", X.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.3, stratify=target, random_state=random_state
    )
    # print("X_train shape:", X_train.shape)
    # print("y_train shape:{}".format(y_train.shape))
    n_components = 90
    pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    # print(f"X_train_pca shape: {X_train_pca}")
    if scale:
        from sklearn import preprocessing

        X_train_pca = preprocessing.scale(X_train_pca)
        X_test_pca = preprocessing.scale(X_test_pca)
    # print(clf)
    # print("X_train_pca", X_train_pca.shape)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    # print("accuracy score:{:.4f}".format(accuracy))
    # print(len(clf.support_x_), len(clf.support_y_))
    # print(clf.support_x_, clf.support_y_)
    assert accuracy >= expected_accuracy


def test_fris_one_class_dataset():
    with pytest.raises(ValueError):
        first_class_points = [(0, 1), (1, 2)]
        second_class_points = []
        third_class_points = []
        points_X, points_y = make_three_class_dataset(first_class_points, second_class_points, third_class_points)
        classifier = FrisStolpWithDistanceCaching()
        classifier.fit(points_X, points_y, True)


def test_fris_same_point_in_two_class_dataset():
    with pytest.raises(ValueError):
        first_class_points = [(0, 1)]
        second_class_points = [(0, 1)]
        third_class_points = []
        points_X, points_y = make_three_class_dataset(first_class_points, second_class_points, third_class_points)
        classifier = FrisStolpWithDistanceCaching()
        classifier.fit(points_X, points_y, True)


def test_fris_duplicated_point_in_same_class_is_ok():
    first_class_points = [(0, 1), (0, 1)]
    second_class_points = [(1, 2)]
    third_class_points = []
    points_X, points_y = make_three_class_dataset(first_class_points, second_class_points, third_class_points)
    classifier = FrisStolpWithDistanceCaching()
    classifier.fit(points_X, points_y, True)
    print(f"{classifier.class_references_=}")
    print(f"{classifier.stolp_elements_indices=}")
    assert len(classifier.class_references_) == 2
    assert classifier.stolp_elements_indices == {0: [0, 1], 2: [2]}


def test_dataset_contains_only_one_example_per_class_two_classes():
    first_class_points = [(0, 1)]
    second_class_points = [(1, 2)]
    third_class_points = []
    points_X, points_y = make_three_class_dataset(first_class_points, second_class_points, third_class_points)
    classifier = FrisStolpWithDistanceCaching()
    classifier.fit(points_X, points_y, True)
    assert len(classifier.class_references_) == 2


def test_dataset_contains_only_one_example_per_class_three_classes():
    first_class_points = [(0, 1)]
    second_class_points = [(1, 2)]
    third_class_points = [(2, 3)]
    points_X, points_y = make_three_class_dataset(first_class_points, second_class_points, third_class_points)
    classifier = FrisStolpWithDistanceCaching()
    classifier.fit(points_X, points_y, True)
    assert len(classifier.class_references_) == 3
    assert len(classifier.classes_) == 3


def test_classification():
    # [(-27.41935483870968, 20.94155844155847, 1), (-5.443548387096776, 4.437229437229462, -1),
    #  (15.524193548387089, 21.48268398268401, -1), (-13.911290322580648, 31.764069264069292, 1),
    #  (-34.4758064516129, 2.813852813852833, 1), (19.153225806451616, -26.677489177489164, -1),
    #  (16.935483870967744, -6.114718614718598, -1), (34.07258064516128, 18.235930735930765, -1),
    #  (11.290322580645153, 36.904761904761926, 1)]
    first_class_points = [
        (-27.41935483870968, 20.94155844155847),
        (-13.911290322580648, 31.764069264069292),
        (-34.4758064516129, 2.813852813852833),
        (11.290322580645153, 36.904761904761926),
    ]
    second_class_points = [
        (-5.443548387096776, 4.437229437229462),
        (15.524193548387089, 21.48268398268401),
        (19.153225806451616, -26.677489177489164),
        (16.935483870967744, -6.114718614718598),
        (34.07258064516128, 18.235930735930765),
    ]
    points_X, points_y = make_three_class_dataset(first_class_points, second_class_points, [])
    print(points_X, points_y)
    clf = FrisStolpWithDistanceCaching()

    clf.fit(points_X, points_y, True)
    print("clf.support_x_indices_", clf.support_x_indices_)
    print("clf.support_x_", np.array(clf.support_x_))
    print("clf.support_y_", clf.support_y_)
    prediction = clf.predict(points_X)
    print(points_X)
    print("points_y  ", points_y)
    print("prediction", prediction)
    print("diff      ", points_y != prediction)
    assert np.array_equal(points_y, prediction)


def test_brute_force_nearest_neighbour_similarity_results():
    points = np.array(
        [
            [-27.41935484, 20.94155844],
            [-13.91129032, 31.76406926],
            [-34.47580645, 2.81385281],
            [11.29032258, 36.9047619],
            [-5.44354839, 4.43722944],
            [15.52419355, 21.48268398],
            [19.15322581, -26.67748918],
            [16.93548387, -6.11471861],
            [34.07258065, 18.23593074],
        ]
    )
    class_references_ = {
        0: np.array([[-13.91129032, 31.76406926], [-34.47580645, 2.81385281], [11.29032258, 36.9047619]]),
        1: np.array(
            [
                [19.15322581, -26.67748918],
                [34.07258065, 18.23593074],
                [-5.44354839, 4.43722944],
                [15.52419355, 21.48268398],
            ]
        ),
    }
    class_negation_references_ = {
        0: np.array(
            [
                [19.15322581, -26.67748918],
                [34.07258065, 18.23593074],
                [-5.44354839, 4.43722944],
                [15.52419355, 21.48268398],
            ]
        ),
        1: np.array([[-13.91129032, 31.76406926], [-34.47580645, 2.81385281], [11.29032258, 36.9047619]]),
    }
    distance = fast_euclidean_distance
    bf_nearest_neighbour_searcher = BruteForceNearestRepresentativeSearcher(
        class_references_, class_negation_references_, distance
    )
    kdtree_nearest_neighbour_searcher = KDTreeNearestRepresentativeSearcher(
        class_references_, class_negation_references_, distance
    )
    classes = [0, 1]

    for p in points:
        bf_distances = bf_nearest_neighbour_searcher.compute_distances_to_nearest_representatives(p, classes)

        kd_distances = kdtree_nearest_neighbour_searcher.compute_distances_to_nearest_representatives(p, classes)
        print("point", p, "bf_distances", bf_distances, "kd_distance", kd_distances)
        assert np.array_equal(bf_distances, kd_distances)


def test_geometric_average():
    assert 1.0 == geometric_average([])


def test_fris_iris():
    run_on_iris(clf=FrisStolpWithDistanceCaching())


def test_fris_predict_proba():
    clf = FrisStolpWithDistanceCaching()
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(y_test)
    for y in probs:
        assert y.shape == (3,)
        assert 1.0 == pytest.approx(y.sum())


def test_corrected_small_problematic():
    first_class_points = [(-38.666, 6.102), (-44.115, -16.700), (-45.641, 15.520)]
    second_class_points = [(10.375, 42.487), (2.964, -25.126), (11.465, -35.040)]
    points_X, points_y = make_three_class_dataset(first_class_points, second_class_points, [])
    print(points_X, points_y)
    clf = FrisStolpWithDistanceCachingCorrected()

    clf.fit(points_X, points_y, True)


def test_on_olivetti_brute():
    run_on_olivetti(
        clf=FrisStolpWithDistanceCaching(
            nearest_reference_searcher_factory=BruteForceNearestRepresentativeSearcher,
            check_for_errors=False,
        ),
        scale=False,
        expected_accuracy=0.85,
    )
    # testing with scaling on data leads to performance improvement,
    # but warning about numerical issues
    # run_on_olivetti(
    #    clf=FrisStolpWithDistanceCaching(
    #        nearest_reference_searcher_factory=BruteForceNearestRepresentativeSearcher,
    #        check_for_errors=False,
    #    ),
    #    scale=True,
    #    expected_accuracy=0.8583333333333333,
    # )


def test_on_olivetti_brute_corrected():
    run_on_olivetti(
        clf=FrisStolpWithDistanceCachingCorrected(
            nearest_reference_searcher_factory=BruteForceNearestRepresentativeSearcher,
        ),
        scale=False,
        expected_accuracy=0.85,
    )
    # testing with scaling on data leads to performance improvement,
    # but warning about numerical issues
    # run_on_olivetti(
    #    clf=FrisStolpWithDistanceCachingCorrected(
    #        nearest_reference_searcher_factory=BruteForceNearestRepresentativeSearcher,
    #    ),
    #    scale=True,
    #    expected_accuracy=0.858333,
    # )
    # run_on_olivetti(
    #    clf=FrisStolpWithDistanceCachingCorrected(
    #        nearest_reference_searcher_factory=BruteForceNearestRepresentativeSearcher,
    #        do_second_round=False,
    #    ),
    #    scale=True,
    #    expected_accuracy=0.858333,
    # )


def test_on_olivetti_KDTree():
    run_on_olivetti(
        clf=FrisStolpWithDistanceCaching(
            nearest_reference_searcher_factory=KDTreeNearestRepresentativeSearcher,
            check_for_errors=False,
        ),
        scale=False,
        expected_accuracy=0.85,
    )
    # testing with scaling on data leads to performance improvement,
    # but warning about numerical issues
    # run_on_olivetti(
    #    clf=FrisStolpWithDistanceCaching(
    #        nearest_reference_searcher_factory=KDTreeNearestRepresentativeSearcher,
    #        check_for_errors=False,
    #    ),
    #    scale=True,
    #    expected_accuracy=0.8583333333333333,
    # )


# Expected Properties for stolp/Fris stolp algorithms
# Any stolp is one of the points of the original dataset
# In case of compact dataset
# Classification with stolps on the original dataset equals original classification
# Points corresponding to stolps correspond to the original dataset
