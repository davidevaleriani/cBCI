import numpy as np

def metacognitive_accuracy(accuracy, confidence):
    """Compute metacognitive accuracy of each trial as:
    100*(1-abs(accuracy-confidence))

    Missing values (e.g, nan) for either accuracy or confidence produce
    np.nan.

    Parameters
    ----------
    accuracy : 1D array or int
        If int, 1 if the decision is correct, 0 otherwise. If array, sequence
        of objective accuracy (1 if correct, 0 otherwise).
    confidence : 1D array or float
        Confidence measure associated to the decision. Should be the same number
        of elements of accuracy and all values in range 0-1.

    Returns
    -------
    1D array or float
        Metacognitive accuracy or array of metacognitive accuracies, in range
        0-100.

    """
    if np.logical_xor(isinstance(accuracy, int), isinstance(confidence, float)):
        raise ValueError("Accuracy should be int, confidence should be float")
    if np.any(np.logical_or(confidence > 1, confidence < 0)):
        raise ValueError("Confidence values should be within range 0-1")
    accuracy = np.array([accuracy]).flatten()
    confidence = np.array([confidence]).flatten()
    if len(accuracy) != len(confidence):
        raise ValueError("Both accuracy and confidence should have the same number of values")
    mac = 100*(1-abs(accuracy-confidence))
    if len(mac) == 1:
        return mac[0]
    return mac
