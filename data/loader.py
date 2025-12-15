"""
DataLoader factory for anomaly detection datasets.

Provides convenient functions to create train and test dataloaders with appropriate
configurations based on dataset format and requirements.
"""

from typing import Tuple, Dict, Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import NpzDataset, MatDataset, CsvDataset, AnomalyDetectionDataset

# Dataset format configurations
MAT_DATASETS = {
    'arrhythmia',
    'wbc'
}

NPZ_DATASETS = {
    'census', 'campaign', 'cardiotocography', 'fraud', 'satellite', 'satimage-2',
    'shuttle', 'breastw', 'cardio', 'thyroid', 'glass',
    'nslkdd', 'optdigits', 'pendigits', 'wine', 'pima', 'ionosphere', 'mammography',
}

# Default dataloader configurations
DEFAULT_DATALOADER_CONFIG = {
    'batch_size': 32,
    'num_workers': 4,
    'shuffle_train': False,
    'shuffle_test': False,
    'pin_memory': True,
    'drop_last': False,
}


def get_dataset_class(dataset_name: str):
    """
    Determine the appropriate dataset class based on dataset name.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Appropriate dataset class (MatDataset, NpzDataset, or CsvDataset)
    """
    if dataset_name in MAT_DATASETS:
        return MatDataset
    elif dataset_name in NPZ_DATASETS:
        return NpzDataset
    else:
        return CsvDataset


def create_dataloader(
        dataset: AnomalyDetectionDataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs
) -> DataLoader:
    """
    Create a DataLoader with specified configuration.

    Args:
        dataset: PyTorch Dataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        **kwargs: Additional arguments to pass to DataLoader

    Returns:
        Configured DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        **kwargs
    )


def get_dataloader(
        model_config: Dict,
        shuffle_train: Optional[bool] = None,
        shuffle_test: Optional[bool] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders based on model configuration.

    Args:
        model_config: Dictionary containing model configuration with keys:
            - dataset_name (str): Name of the dataset
            - data_dim (int): Number of feature dimensions
            - data_dir (str): Directory containing dataset files
            - batch_size (int, optional): Batch size for dataloaders
            - num_workers (int, optional): Number of worker processes
            - shuffle_train (bool, optional): Whether to shuffle training data
            - shuffle_test (bool, optional): Whether to shuffle test data
            - pin_memory (bool, optional): Whether to pin memory
            - drop_last (bool, optional): Whether to drop last incomplete batch
        shuffle_train: Override shuffle setting for training loader
        shuffle_test: Override shuffle setting for test loader

    Returns:
        Tuple of (train_loader, test_loader)

    Raises:
        KeyError: If required configuration keys are missing
        FileNotFoundError: If dataset files are not found
        ValueError: If configuration values are invalid

    Example:
        >>> config = {
        ...     'dataset_name': 'thyroid',
        ...     'data_dim': 6,
        ...     'data_dir': './datasets',
        ...     'batch_size': 64,
        ...     'num_workers': 4
        ... }
        >>> train_loader, test_loader = get_dataloader(config)
        >>> print(f"Train batches: {len(train_loader)}")
    """
    # Validate required configuration keys
    required_keys = ['dataset_name', 'data_dim', 'data_dir']
    missing_keys = [key for key in required_keys if key not in model_config]

    if missing_keys:
        raise KeyError(f"Missing required configuration keys: {missing_keys}")

    # Extract configuration with defaults
    dataset_name = model_config['dataset_name']
    data_dim = model_config['data_dim']
    data_dir = model_config['data_dir']

    # Validate data directory exists
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # DataLoader configurations with defaults
    batch_size = model_config.get('batch_size', DEFAULT_DATALOADER_CONFIG['batch_size'])
    num_workers = model_config.get('num_workers', DEFAULT_DATALOADER_CONFIG['num_workers'])
    pin_memory = model_config.get('pin_memory', DEFAULT_DATALOADER_CONFIG['pin_memory'])
    drop_last = model_config.get('drop_last', DEFAULT_DATALOADER_CONFIG['drop_last'])

    # Shuffle configurations
    if shuffle_train is None:
        shuffle_train = model_config.get('shuffle_train', DEFAULT_DATALOADER_CONFIG['shuffle_train'])
    if shuffle_test is None:
        shuffle_test = model_config.get('shuffle_test', DEFAULT_DATALOADER_CONFIG['shuffle_test'])

    # Validate batch size
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # Determine dataset class
    dataset_class = get_dataset_class(dataset_name)

    # Create datasets
    try:
        train_set = dataset_class(
            dataset_name=dataset_name,
            data_dim=data_dim,
            data_dir=data_dir,
            mode='train'
        )
        test_set = dataset_class(
            dataset_name=dataset_name,
            data_dim=data_dim,
            data_dir=data_dir,
            mode='test'
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")

    # Create dataloaders
    train_loader = create_dataloader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    test_loader = create_dataloader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Never drop last batch for test set
    )

    return train_loader, test_loader


def get_dataloader_auto(
        dataset_name: str,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Automatically detect dataset format and create dataloaders.

    This function automatically determines the appropriate dataset class
    by checking which file format exists in the data directory.

    Args:
        dataset_name: Name of the dataset (without extension)
        data_dir: Directory containing dataset files
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        **kwargs: Additional arguments for dataloader configuration

    Returns:
        Tuple of (train_loader, test_loader)

    Example:
        >>> train_loader, test_loader = get_dataloader_auto(
        ...     'thyroid',
        ...     './datasets',
        ...     batch_size=64
        ... )
    """
    data_path = Path(data_dir)

    # Determine data dimension by loading a sample
    # This is a workaround since we don't have data_dim
    data_dim = None

    # Try to detect the file and get dimensions
    for ext, dataset_class in [('.mat', MatDataset),
                               ('.npz', NpzDataset),
                               ('.csv', CsvDataset)]:
        if (data_path / f"{dataset_name}{ext}").exists():
            # Create a temporary dataset to get dimensions
            temp_dataset = dataset_class(
                dataset_name=dataset_name,
                data_dim=0,  # Placeholder
                data_dir=data_dir,
                mode='train'
            )
            data_dim = temp_dataset.data.shape[1]
            break

    if data_dim is None:
        raise FileNotFoundError(
            f"No dataset file found for '{dataset_name}' in {data_dir}"
        )

    # Create configuration and use main function
    config = {
        'dataset_name': dataset_name,
        'data_dim': data_dim,
        'data_dir': data_dir,
        'batch_size': batch_size,
        'num_workers': num_workers,
        **kwargs
    }

    return get_dataloader(config)


def get_dataset_info(dataset_name: str) -> Dict[str, str]:
    """
    Get information about a dataset's expected format.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dictionary with format information

    Example:
        >>> info = get_dataset_info('thyroid')
        >>> print(info['format'])
        'mat'
    """
    if dataset_name in MAT_DATASETS:
        return {'format': 'mat', 'class': 'MatDataset'}
    elif dataset_name in NPZ_DATASETS:
        return {'format': 'npz', 'class': 'NpzDataset'}
    else:
        return {'format': 'csv', 'class': 'CsvDataset'}
