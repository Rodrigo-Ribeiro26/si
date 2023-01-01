import numpy as np

def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    It returns the derivative of the mean squared error for the y_pred variable.
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
    Returns:
        np.ndarray: Derivative of the mean squared error
    """
    return -2 * (y_true - y_pred) / (len(y_true) * 2)