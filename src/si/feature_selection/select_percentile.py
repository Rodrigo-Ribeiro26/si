import sys
sys.path.append("C:\\Users\\rodri\\Desktop\\Mestrado\\SIB\\si\\src\\si")
    
from typing import Callable
from statistics_.f_classification import f_classification
from data.dataset import Dataset
import numpy as np


class SelectPercentile:
    
    def __init__(self, percentile: float = 0.25, score_func: Callable = f_classification) -> None:
        """Select the highest scoring features according to the percentile given.
            The scores are computed using ANOVA F-values between label/feature.
        Args:
            percentile (float, optional): Percentile of features to select from all the features. Defaults to 0.25.
            score_func (Callable, optional): Function that takes a dataset and returns the scores and p-values. Defaults to f_classification.
        Raises:
            ValueError: Raises ValueError if percentile isn't between 0 and 1.
        """
        
        if percentile > 1 or percentile < 0:
            raise ValueError("Your percentile must be a float between 0 and 1")
        
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None
        
    def fit(self, dataset: Dataset) -> "SelectPercentile":
        """Fits SelectPercentile to compute the F scores and p_values.
        Args:
            dataset (Dataset): A dataset object
        Returns:
            SelectKBest: returns self
        """
        self.F, self.p = self.score_func(dataset)
        return self
    
    def transform(self, dataset: Dataset) -> Dataset:
        """It transforms the dataset by selecting the highest scores according to the percentile.
        Args:
            dataset (Dataset): A dataset object
        Returns:
            Dataset: A dataset object with the percentile highest score features.
        """
        len_features = len(dataset.features)
        percentile = int(len_features * self.percentile)
        idxs = np.argsort(self.F)[-percentile:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)
    
    def fit_transform(self, dataset: Dataset) -> Dataset:
        """It runs the fit and the transform methods of this class automatically.
        Args:
            dataset (Dataset): A dataset object
        Returns:
            Dataset: A dataset object with the percentile highest score features.
        """
        self.fit(dataset)
        return self.transform(dataset)
    
    

if __name__ == "__main__":
    a = SelectPercentile(0.75)
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    a = a.fit_transform(dataset)
    print(dataset.features)
    print(a.features)  