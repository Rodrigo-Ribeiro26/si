import sys
sys.path.insert(0,"C:/Users/rodri/Desktop/Mestrado/SIB/si/src")

from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate
import random
from typing import Dict, Callable, List, Optional, Union

def randomized_search_cv(model, dataset: Dataset, parameter_distribution: Union[Dict, List[Dict]], scoring: Callable = None, cv: int = 3, test_size: float = 0.2, n_iter: int = 10):
    
    for parameter in parameter_distribution:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}")
        
    scores = {
        "parameters": [],
        "seeds": [],
        "train": [],
        "test": []
    }
    
    for i in range(n_iter):
        # set the random seed and append it
        random_seed_state = random.randint(0, 1000)
        scores["seeds"].append(random_seed_state)
        
        # dictionary for parameters config and parameter selection and setting on model
        parameters = {}
        
        for parameter, value in parameter_distribution.items():
            parameters[parameter] = random.choice(value)
            
        for parameter, value in parameters.items():
            setattr(model, parameter, value)
            
        # get scores from cross validation
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)
        
        # append everything to dictionary for return
        scores["parameters"].append(parameters)
        scores["train"].append(score["train"])
        scores["test"].append(score["test"])
        
    return scores