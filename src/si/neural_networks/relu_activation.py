import numpy as np

class ReLUActivation:
    def __init__(self) -> None:
        self.input_data = None
        
    def forward(self, input_data: np.ndarray):
        """
        Computes the rectified linear unit relationship
        Args:
            input_data (np.ndarray): Input data
        """
        self.input_data = input_data
        
        return np.maximum(0, input_data) 
    
    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Computes teh backward pass of the rectified linear unit relationshop.
        Args:
            error (np.ndarray): Error
            learning_rate(float): Learning rate
        Returns:
            np.ndarray: Error of the previous layer.
        """
        
        relu_deriv = np.where(self.input_data > 0, 1, 0)
        
        error_to_propagate = error * relu_deriv
        
        return error_to_propagate