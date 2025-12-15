import numpy as np
from typing import Dict

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.5,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    - Training: Half of normal data
    - Test: Rest of normal data + all anomalies
    """
    np.random.seed(random_state)

    normal_mask = y == 0
    anomaly_mask = y == 1

    X_normal = X[normal_mask]
    X_anomaly = X[anomaly_mask]

    idx = np.random.permutation(len(X_normal))
    X_normal = X_normal[idx]

    # Split
    n_train = int(len(X_normal) * (1 - test_ratio))

    X_train = X_normal[:n_train]
    X_test_normal = X_normal[n_train:]

    # Test set = normal test + all anomalies
    X_test = np.vstack([X_test_normal, X_anomaly])
    y_test = np.concatenate([
        np.zeros(len(X_test_normal)),
        np.ones(len(X_anomaly))
    ])

    # Shuffle test
    idx = np.random.permutation(len(X_test))
    X_test = X_test[idx]
    y_test = y_test[idx]

    return {
        'X_train': X_train,
        'y_train': np.zeros(len(X_train)),
        'X_test': X_test,
        'y_test': y_test
    }