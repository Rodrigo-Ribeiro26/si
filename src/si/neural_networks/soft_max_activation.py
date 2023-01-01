import numpy as np


class SoftMaxActivation:
    def __init__(self):
        pass
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Computes the probability of each class.
        Args:
            input_data (np.ndarray): Input data
        Returns:
            np.ndarray: The probability of each class
        """
        
        ziexp = np.exp(input_data - np.max(input_data))
        return (ziexp / (np.sum(ziexp, axis=1, keepdims=True))) # sum by row and keep the dimensions of the array