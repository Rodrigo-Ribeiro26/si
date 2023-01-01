import sys
sys.path.insert(0,"C:/Users/rodri/Desktop/Mestrado/SIB/si/src")

import numpy as np
from si.statistics_.sigmoid_function import sigmoid_function

class SigmoidActivation:
    def __init__(self):
        # attributes
        self.input_data = None
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the layer.
        Args:
            input_data (np.ndarray): Input data.
        """
        self.input_data = input_data
        return sigmoid_function(input_data)
    
    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Computes the backward pass of the layer.
        Args:
            error (np.ndarray): Error
        Returns:
            np.ndarray: Return the error of the previous layer.
        """
        
        # multiplication by the derivative and not the entire matrix
        sigmoid_deriv = sigmoid_function(self.input_data) * (1 - sigmoid_function(self.input_data))
        
        error_to_propagate = error * sigmoid_deriv
        
        return 