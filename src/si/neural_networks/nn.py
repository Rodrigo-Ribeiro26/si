import sys
sys.path.insert(0,"C:/Users/rodri/Desktop/Mestrado/SIB/si/src")
from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.metrics.mse import mse, mse_derivative


import numpy as np
from typing import Callable

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.metrics.mse_derivative import mse_derivative
from si.metrics.mse import mse


class NN:
    def __init__(self, 
                 layers: list, 
                 epochs: int = 1000, 
                 learning_rate: float = 0.01,
                 loss: Callable = mse,
                 loss_derivative: Callable = mse_derivative,
                 verbose: bool = False):
        """
        The NN is the Neural Network class model.
        
        Args:
            layers (list): List of layers in the neural network.
            epochs (int, optional): Number of epochs to train the model. Defaults to 1000.
            learning_rate (float, optional): Learning rate of our model. Defaults to 0.01.
            loss (Callable, optional): Loss function to use. Defaults to mse.
            loss_derivative (Callable, optional): The derivative of the loss function to use. Defaults to mse_derivative.
            verbose (bool, optional): Whether to print loss int ate each epoch. Defaults to False.
        """
        # parameters
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss
        self.loss_derivative = loss_derivative
        self.verbose = verbose
        
        # attributes
        self.history = {} # records the history of our model training
        
        
    def fit(self, dataset: Dataset) -> "NN":
        """
        It fits our model to the given dataset
        Args:
            dataset (Dataset): Dataset to fit the model to
        Returns:
            NN: Fitted model
        """
        
        for epoch in range(1, self.epochs + 1):
            y_pred = dataset.X
            y_true = np.reshape(dataset.y, (-1, 1))
            
            # forward propagation
            for layer in self.layers:
                y_pred = layer.forward(y_pred)
                
            # backward propagation
            error = self.loss_derivative(y_true, y_pred)
            for layer in self.layers[::-1]:
                error = layer.backward(error, self.learning_rate)
                
            # save history
            cost = self.loss_function(y_true, y_pred)
            self.history[epoch] = cost
            
            # print loss if verbose = True
            if self.verbose:
                print(f'Epoch {epoch}/{self.epochs} - cost: {cost}')
                
        return self
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the output of the given dataset.
        Args:
            dataset (Dataset): The dataset to predict the output
        Returns:
            np.ndarray: The predicted output
        """
        X = dataset.X
        
        # forward propagation
        for layer in self.layers:
            X = layer.forward(X)
            
        return X
    
    def cost(self, dataset: Dataset) -> float:
        """
        It computes the cost of the model in the given dataset
        Args:
            dataset (Dataset): The dataset to compute the cost
        Returns:
            float: The cost of the model
        """
        y_pred = self.predict(dataset)
        return self.loss_function(dataset.y, y_pred)
    
    def score(self, dataset: Dataset, scoring_function: Callable = accuracy) -> float:
        """
        It computes the score of our model.
        Args:
            dataset (Dataset): The dataset used to compute the score
            scoring_function (Callable, optional): The scoring function that will be used to compute score. Defaults to accuracy.
        Returns:
            float: The score of our model
        """
        y_pred = self.predict(dataset)
        return scoring_function(dataset.y, y_pred)
