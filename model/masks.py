import numpy as np


class MaskingStrategies:
    @staticmethod
    def matrix_masking(X: np.ndarray, random_state: int = 42) -> np.ndarray:
        """
        Matrix Masking
        """
        np.random.seed(random_state)
        M = np.random.uniform(0, 1, size=X.shape)
        return X * M

    @staticmethod
    def zero_masking(X: np.ndarray, mask_prob: float = 0.4,
                     random_state: int = 42) -> np.ndarray:
        """
        Zero Masking
        """
        np.random.seed(random_state)
        mask = np.random.binomial(1, 1 - mask_prob, size=X.shape)
        return X * mask

    @staticmethod
    def transformation_masking(X: np.ndarray, random_state: int = 42) -> np.ndarray:
        """
        Transformation Masking
        """
        np.random.seed(random_state)
        W = np.random.randn(X.shape[1], X.shape[1])
        return X @ W

    @staticmethod
    def shuffle_masking(X: np.ndarray, mask_prob: float = 0.4,
                        random_state: int = 42) -> np.ndarray:
        """
        Shuffle Masking
        """
        np.random.seed(random_state)
        X_masked = X.copy()

        for j in range(X.shape[1]):
            mask = np.random.binomial(1, mask_prob, size=X.shape[0])
            replacement = np.random.choice(X[:, j], size=mask.sum())
            X_masked[mask == 1, j] = replacement

        return X_masked
