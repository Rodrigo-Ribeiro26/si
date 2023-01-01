import numpy as np

class LinearActivation:
    def __init__(self) -> None:
        pass
    
    def forward(input_data: np.ndarray) -> np.ndarray:
        """
        Computes the linear activation, also known as "No activation".
        :param input_data: input data
        :return: Returns the input data. The linear activation basically spits out the input data as it is.
        """
        
        return input_data