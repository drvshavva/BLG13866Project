import numpy as np
from typing import Tuple


def generate_synthetic_data(
        n_normal: int = 1000,
        n_anomaly: int = 50,
        n_features: int = 10,
        anomaly_type: str = 'global',
        random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset with different types of anomalies.

    4 type anomaly:
    1. Local: Normal distribution small differance
    2. Global: Far from normal data
    3. Dependency: Break correlation between features
    4. Clustered: Create a cluster of anomalies
    """
    np.random.seed(random_state)

    # Normal data - correlated features
    mean = np.zeros(n_features)

    # Covariance matrix with correlations
    cov = np.eye(n_features)
    cov[0, 1] = cov[1, 0] = 0.7  # Feature 0-1 correlation
    cov[2, 3] = cov[3, 2] = 0.6  # Feature 2-3 correlation

    X_normal = np.random.multivariate_normal(mean, cov, n_normal)

    if anomaly_type == 'global':
        X_anomaly = np.random.randn(n_anomaly, n_features) * 2 + 5

    elif anomaly_type == 'local':
        X_anomaly = np.random.multivariate_normal(mean, cov * 3, n_anomaly)

    elif anomaly_type == 'dependency':
        X_anomaly = np.random.randn(n_anomaly, n_features)

    elif anomaly_type == 'clustered':
        cluster_mean = np.ones(n_features) * 4
        X_anomaly = np.random.multivariate_normal(cluster_mean, cov * 0.5, n_anomaly)
    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

    return X, y
