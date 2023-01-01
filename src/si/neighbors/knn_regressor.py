import sys
sys.path.insert(0,"C:/Users/rodri/Desktop/Mestrado/SIB/si/src")

import numpy as np
from typing import Callable
from si.statistics_.euclidean_distance import euclidean_distance
from si.data.dataset import Dataset
from si.metrics.rmse import rmse

class KNNRegressor:
    
    def __init__(self, k: int, distance: Callable = euclidean_distance) -> None:
        """
        This algorithm predicts the class for a sample using the k most similar examples.
        Args:
            k (int): number of examples to consider
            distance (Callable, optional): euclidean distance function. Defaults to euclidean_distance.
        """
        self.k = k
        self.distance = distance
        self.dataset = None
        
    def fit(self, dataset: Dataset) -> "KNNRegressor":
        self.dataset = dataset # dataset de treino para o modelo
        return self
    
    def _get_closest_label(self, sample):
        
        # Calculates the distance between the samples and the dataset
        distances = self.distance(sample, self.dataset.X)
        
        # sort dinstances get indexes
        k_nearest_neighbor = np.argsort(distances)[:self.k]
        k_nearest_neighbor_label = self.dataset.y[k_nearest_neighbor]
        
        return np.mean(k_nearest_neighbor_label)
    
    def predict(self, dataset: Dataset):
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
        
    def score(self, dataset: Dataset) -> float:
        prediction = self.predict(dataset)
        return rmse(dataset.y, prediction)