import numpy as np
from typing import List


def metacognitive_accuracy(accuracy: List, confidence: List) -> List:
    """Compute metacognitive accuracy of each trial as:
    100*(1-abs(accuracy-confidence))

    Missing values (e.g, nan) for either accuracy or confidence produce
    np.nan.

    Parameters
    ----------
    accuracy : list
        Sequence of objective accuracy (1 if correct, 0 otherwise) of each trial.
    confidence : list
        Confidence measure associated to the decision. Should be the same number
        of elements of accuracy and all values in range 0-1.

    Returns
    -------
    list
        Sequence of metacognitive accuracies for each trial, in range 0-100.

    """
    if np.logical_xor(isinstance(accuracy, int), isinstance(confidence, float)):
        raise ValueError("Accuracy should be int, confidence should be float")
    if np.any(np.logical_or(confidence > 1, confidence < 0)):
        raise ValueError("Confidence values should be within range 0-1")
    accuracy = np.array(accuracy)
    confidence = np.array(confidence)
    if len(accuracy) != len(confidence):
        raise ValueError("Both accuracy and confidence should have the same number of values")
    # Calculate metacognitive accuracy
    mac = 100*(1-abs(accuracy-confidence))
    return mac
