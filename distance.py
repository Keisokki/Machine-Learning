import numpy as np
import pandas as pd

class DistanceModel:

    def __init__(self, k: int = 3) -> None:
        self.k = k
    
    def fit(self, X_train: pd.core.frame.DataFrame, y_train: pd.core.series.Series) -> None:
        self.X_train = X_train.values
        self.y_train = y_train.values

    def k_nearest_indices(self, single_X_test: pd.core.series.Series) -> np.ndarray:
        distances = np.array([np.linalg.norm(single_X_test-x) for x in self.X_train])
        k_nearest = distances.argsort()[:self.k] # Nyari foods yang mirip-mirip
        return k_nearest