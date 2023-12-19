import numpy as np
from cBCI.groups.performance import majority


def test_standard_majority():
    """ Test standard majority for 3 users."""
    decisions = np.array([[1, 1],
                          [1, 1],
                          [0, 0]])
    correctness = np.array([1, 1])
    accuracy = majority(decisions, correctness, tie_policy='random_balanced')
    assert len(accuracy) == decisions.shape[0]
    assert accuracy[0] == [1, 1, 0]
    assert accuracy[1] == [1, 0.5, 0.5]
    assert accuracy[2] == [1]


def test_standard_majority_with_nans():
    """ Test standard majority for 3 users with nan decisions."""
    decisions = np.array([[1, 1],
                          [1, 1],
                          [np.nan, 0]])
    correctness = np.array([1, 1])
    accuracy = majority(decisions, correctness, tie_policy='random_balanced')
    assert accuracy == 0
    assert len(accuracy) == decisions.shape[0]
    assert accuracy[0] == [1, 1, 0]
    assert accuracy[1] == [1, 0.75, 0.75]
    assert accuracy[2] == [1]
