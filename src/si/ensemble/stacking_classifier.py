import sys
sys.path.insert(0,"C:/Users/rodri/Desktop/Mestrado/SIB/si/src")

from typing import List
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
import numpy as np

class StackingClassifier:
    """
    Uses a set of models to predict the outcome. These predictions are used to train a final model which is then used to
    predict the final outcome of the output variable (Y)
    """
    def __init__(self, models: List[object], final_model: object) -> None:
        """Initializes the StackingClassifier class
        Args:
            models (List[object]): A list of models for the ensemble
            final_model (object): Final model to use the predictions of the other models and make the final prediction
        """
        self.models = models
        self.final_model = final_model
        
        
    def fit(self, dataset: Dataset) -> "StackingClassifier":
        """Fits the model according to the given training dataset
        Args:
            dataset (Dataset): Training Dataset object
        Returns:
            StackingClassifier: self
        """
        
        # trains the models
        for model in self.models:
            model.fit(dataset)
            
        # get predictions for each model
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))
            
        # trains the final model with the predictions from the models
        self.final_model.fit(Dataset(dataset.X, np.array(predictions).T))
        
        return self
    
    def predict(self, dataset: Dataset) -> np.array:
        """Collects the predictions of all the models and computes the final prediction of the final model returning it.
        Args:
            dataset (Dataset): Dataset object
        Returns:
            np.array: Final model prediction
        """
        # models predictions
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))
            
        # gets the final prediction
        y_pred_final = self.final_model.predict(Dataset(dataset.X, np.array(predictions).T))
        
        return y_pred_final
    
    def score(self, dataset: Dataset) -> float:
        """Computes the score of the model by calculating the accuracy metric
        Args:
            dataset (Dataset): Dataset object
        Returns:
            float: Accuracy of the model
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)