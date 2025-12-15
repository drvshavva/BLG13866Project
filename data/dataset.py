"""
PyTorch Dataset classes for loading anomaly detection datasets from various file formats.

Supports CSV, MAT (MATLAB), and NPZ (NumPy) file formats with automatic train/test splitting
based on normal (inlier) and anomalous (outlier) samples.
"""

import os
from pathlib import Path
from typing import Tuple, Literal

import csv
import numpy as np
import torch
from scipy import io
from torch.utils.data import Dataset


class AnomalyDetectionDataset(Dataset):
    """
    Base class for anomaly detection datasets with common functionality.

    This abstract base class provides common methods for train/test mode handling
    and implements the standard PyTorch Dataset interface.
    """

    def __init__(self, mode: Literal['train', 'test'] = 'train'):
        """
        Initialize the dataset.

        Args:
            mode: Either 'train' or 'test' to determine which split to load
        """
        super().__init__()
        self.mode = mode
        self.data: torch.Tensor = None
        self.targets: torch.Tensor = None

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample and its label.

        Args:
            index: Index of the sample to retrieve

        Returns:
            Tuple of (sample, label)
        """
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def _load_and_split(
            self,
            samples: np.ndarray,
            labels: np.ndarray
    ) -> None:
        """
        Split data into train/test sets and convert to tensors.

        Args:
            samples: Feature array of shape (n_samples, n_features)
            labels: Binary labels (0: normal, 1: anomaly)
        """
        # Separate inliers and outliers
        inliers = samples[labels == 0]
        outliers = samples[labels == 1]

        # Split data
        train_data, train_labels, test_data, test_labels = split_train_test(
            inliers, outliers
        )

        # Assign to appropriate split based on mode
        if self.mode == 'train':
            self.data = torch.FloatTensor(train_data)
            self.targets = torch.FloatTensor(train_labels)
        else:
            self.data = torch.FloatTensor(test_data)
            self.targets = torch.FloatTensor(test_labels)

        print(f"Loaded {self.mode} dataset with {len(self.data)} samples")


class CsvDataset(AnomalyDetectionDataset):
    """
    Dataset loader for CSV files containing anomaly detection data.

    Expected CSV format:
        - First `data_dim` columns: features
        - Last column: binary label (0: normal, 1: anomaly)
    """

    def __init__(
            self,
            dataset_name: str,
            data_dim: int,
            data_dir: str,
            mode: Literal['train', 'test'] = 'train'
    ):
        """
        Initialize CSV dataset.

        Args:
            dataset_name: Name of the dataset (without .csv extension)
            data_dim: Number of feature dimensions
            data_dir: Directory containing the dataset file
            mode: Either 'train' or 'test'

        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If data cannot be properly parsed
        """
        super().__init__(mode)

        file_path = Path(data_dir) / f"{dataset_name}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load data from CSV
        samples = []
        labels = []

        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                samples.append(row[:data_dim])
                labels.append(row[data_dim])

        # Convert to numpy arrays
        samples = np.array(samples, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        # Split and load
        self._load_and_split(samples, labels)


class MatDataset(AnomalyDetectionDataset):
    """
    Dataset loader for MATLAB .mat files containing anomaly detection data.

    Expected .mat format:
        - 'X': Feature matrix of shape (n_samples, n_features)
        - 'y': Label vector of shape (n_samples, 1) with values {0, 1}
    """

    def __init__(
            self,
            dataset_name: str,
            data_dim: int,
            data_dir: str,
            mode: Literal['train', 'test'] = 'train'
    ):
        """
        Initialize MAT dataset.

        Args:
            dataset_name: Name of the dataset (without .mat extension)
            data_dim: Number of feature dimensions (unused but kept for API consistency)
            data_dir: Directory containing the dataset file
            mode: Either 'train' or 'test'

        Raises:
            FileNotFoundError: If the MAT file doesn't exist
            KeyError: If required keys 'X' or 'y' are not in the .mat file
        """
        super().__init__(mode)

        file_path = Path(data_dir) / f"{dataset_name}.mat"

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load MATLAB file
        mat_data = io.loadmat(str(file_path))

        try:
            samples = mat_data['X']
            labels = mat_data['y'].astype(int).reshape(-1)
        except KeyError as e:
            raise KeyError(f"Required key not found in .mat file: {e}")

        # Split and load
        self._load_and_split(samples, labels)


class NpzDataset(AnomalyDetectionDataset):
    """
    Dataset loader for NumPy .npz files containing anomaly detection data.

    Expected .npz format:
        - 'X': Feature matrix of shape (n_samples, n_features)
        - 'y': Label vector of shape (n_samples,) with values {0, 1}
    """

    def __init__(
            self,
            dataset_name: str,
            data_dim: int,
            data_dir: str,
            mode: Literal['train', 'test'] = 'train'
    ):
        """
        Initialize NPZ dataset.

        Args:
            dataset_name: Name of the dataset (without .npz extension)
            data_dim: Number of feature dimensions (unused but kept for API consistency)
            data_dir: Directory containing the dataset file
            mode: Either 'train' or 'test'

        Raises:
            FileNotFoundError: If the NPZ file doesn't exist
            KeyError: If required keys 'X' or 'y' are not in the .npz file
        """
        super().__init__(mode)

        file_path = Path(data_dir) / f"{dataset_name}.npz"

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load NPZ file
        npz_data = np.load(str(file_path))

        try:
            samples = npz_data['X']
            labels = npz_data['y'].astype(int).reshape(-1)
        except KeyError as e:
            raise KeyError(f"Required key not found in .npz file: {e}")

        # Split and load
        self._load_and_split(samples, labels)


def split_train_test(
        inliers: np.ndarray,
        outliers: np.ndarray,
        train_ratio: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split anomaly detection data into train and test sets.

    Training set contains only normal (inlier) samples.
    Test set contains both normal and anomalous (outlier) samples.

    Strategy:
        - Training: First 50% of inliers (labeled as 0)
        - Testing: Remaining 50% of inliers + all outliers (properly labeled)

    Args:
        inliers: Normal samples of shape (n_inliers, n_features)
        outliers: Anomalous samples of shape (n_outliers, n_features)
        train_ratio: Proportion of inliers to use for training (default: 0.5)

    Returns:
        Tuple containing:
            - train_data: Training samples (only inliers)
            - train_labels: Training labels (all zeros)
            - test_data: Test samples (inliers + outliers)
            - test_labels: Test labels (0 for inliers, 1 for outliers)

    Example:
        >>> inliers = np.random.randn(100, 10)
        >>> outliers = np.random.randn(20, 10)
        >>> train_x, train_y, test_x, test_y = split_train_test(inliers, outliers)
        >>> print(train_x.shape, test_x.shape)
        (50, 10) (70, 10)
    """
    # Calculate split point
    num_train = int(len(inliers) * train_ratio)

    # Create training set (only normal samples)
    train_data = inliers[:num_train]
    train_labels = np.zeros(num_train, dtype=np.float32)

    # Create test set (remaining inliers + all outliers)
    test_data = np.concatenate([inliers[num_train:], outliers], axis=0)

    # Create test labels
    num_test_inliers = len(inliers) - num_train
    test_labels = np.zeros(len(test_data), dtype=np.float32)
    test_labels[num_test_inliers:] = 1  # Mark outliers as anomalies

    return train_data, train_labels, test_data, test_labels


def load_dataset(
        dataset_name: str,
        data_dim: int,
        data_dir: str,
        mode: Literal['train', 'test'] = 'train'
) -> AnomalyDetectionDataset:
    """
    Factory function to load appropriate dataset based on file extension.

    Args:
        dataset_name: Name of the dataset (with or without extension)
        data_dim: Number of feature dimensions
        data_dir: Directory containing the dataset file
        mode: Either 'train' or 'test'

    Returns:
        Appropriate dataset instance (CsvDataset, MatDataset, or NpzDataset)

    Raises:
        ValueError: If file format is not supported

    Example:
        >>> dataset = load_dataset('thyroid', 6, './data', mode='train')
        >>> print(len(dataset))
        1794
    """
    # Remove extension if present
    dataset_base = dataset_name.split('.')[0]

    # Check which file exists
    data_path = Path(data_dir)

    if (data_path / f"{dataset_base}.csv").exists():
        return CsvDataset(dataset_base, data_dim, data_dir, mode)
    elif (data_path / f"{dataset_base}.mat").exists():
        return MatDataset(dataset_base, data_dim, data_dir, mode)
    elif (data_path / f"{dataset_base}.npz").exists():
        return NpzDataset(dataset_base, data_dim, data_dir, mode)
    else:
        raise ValueError(
            f"No supported dataset file found for '{dataset_base}' in {data_dir}. "
            f"Supported formats: .csv, .mat, .npz"
        )
