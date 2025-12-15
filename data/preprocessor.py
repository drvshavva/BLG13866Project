import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    """
    Normalization and standardization operation.

    Args:
        method: 'standard' (z-score) or 'minmax' (0-1 aralığı)
    """

    def __init__(self, method: str = 'standard'):
        self.method = method

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown method: {method}")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(X)

