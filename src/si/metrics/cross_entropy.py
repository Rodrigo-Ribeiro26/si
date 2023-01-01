import numpy as np


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Returns the cross-entropy loss function which is the difference of two probabilities.
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
    Returns:
        float: Difference of the two probabilities
    """
    return - np.sum(y_true) * np.log(y_pred) / len(y_true)


def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ 
    Returns the derivative of the cross-entropy loss function.
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
    Returns:
        float: The derivative of the cross entropy loss function.
    """

    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred)) / len(y_true)