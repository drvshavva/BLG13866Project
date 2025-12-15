"""
Utility functions for anomaly detection model training and evaluation.

This module provides functions for data loading, performance metrics calculation,
result logging, and model weight analysis.
"""

import csv
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support
)


def load_data(file_path: str, num_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load feature data and labels from a CSV file.

    Args:
        file_path: Path to the CSV file containing features and labels
        num_features: Number of feature columns (label is expected in the next column)

    Returns:
        Tuple containing:
            - features: numpy array of shape (n_samples, num_features)
            - labels: numpy array of shape (n_samples,)

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If data cannot be converted to float
    """
    features = []
    labels = []

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            features.append(row[:num_features])
            labels.append(row[num_features])

    # Convert to numpy arrays with proper dtype
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    return features, labels


def calculate_auc_metrics(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Calculate ROC AUC and Average Precision scores.

    Args:
        scores: Anomaly scores predicted by the model
        labels: Ground truth binary labels (0: normal, 1: anomaly)

    Returns:
        Tuple containing:
            - roc_auc: Area Under the ROC Curve
            - avg_precision: Average Precision score
    """
    roc_auc = roc_auc_score(labels, scores)
    avg_precision = average_precision_score(labels, scores)

    return roc_auc, avg_precision


def calculate_f1_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate F1 score using a threshold based on the normal data ratio.

    The threshold is set at the percentile corresponding to the proportion
    of normal samples, effectively treating samples above this threshold as anomalies.

    Args:
        scores: Anomaly scores predicted by the model
        labels: Ground truth binary labels (0: normal, 1: anomaly)

    Returns:
        F1 score for binary classification
    """
    normal_ratio = (labels == 0).sum() / len(labels)
    scores = np.squeeze(scores)

    # Set threshold at the percentile corresponding to normal ratio
    threshold = np.percentile(scores, 100 * normal_ratio)

    # Create binary predictions
    predictions = (scores > threshold).astype(int)

    # Calculate precision, recall, and F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )

    return f1


def write_results(
        model_name: str,
        avg_auc_roc: float,
        avg_auc_pr: float,
        std_auc_roc: float,
        std_auc_pr: float,
        output_path: str
) -> None:
    """
    Append evaluation results to a CSV file.

    Args:
        model_name: Name of the model being evaluated
        avg_auc_roc: Average ROC AUC across folds/runs
        avg_auc_pr: Average Precision-Recall AUC across folds/runs
        std_auc_roc: Standard deviation of ROC AUC
        std_auc_pr: Standard deviation of PR AUC
        output_path: Path to the output CSV file
    """
    with open(output_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            model_name,
            f"{avg_auc_roc:.4f}",
            f"{avg_auc_pr:.4f}",
            f"{std_auc_roc:.4f}",
            f"{std_auc_pr:.4f}"
        ])


def write_attention_weights(
        filename: str,
        test_labels: np.ndarray,
        feature_attention_weights,
        gradient_attention_weights,
        anomaly_scores: np.ndarray,
        num_samples: int = 300,
        output_dir: str = './att_weights'
) -> None:
    """
    Write attention weights to CSV files for analysis.

    Saves two CSV files containing feature-level and gradient-level attention
    weights along with labels and anomaly scores for the first num_samples.

    Args:
        filename: Base name for output files
        test_labels: Ground truth labels for test samples
        feature_attention_weights: Feature attention weights (tensor)
        gradient_attention_weights: Gradient attention weights (tensor)
        anomaly_scores: Predicted anomaly scores
        num_samples: Number of samples to write (default: 300)
        output_dir: Directory to save attention weight files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    feature_path = Path(output_dir) / f"{filename}_fea.csv"
    gradient_path = Path(output_dir) / f"{filename}_grad.csv"

    # Extract and convert attention weights
    feature_weights = feature_attention_weights.detach().cpu().numpy()[:num_samples]
    gradient_weights = gradient_attention_weights.detach().cpu().numpy()[:num_samples]

    # Combine with labels and scores
    feature_data = np.column_stack((
        test_labels[:num_samples],
        anomaly_scores[:num_samples],
        feature_weights
    ))

    gradient_data = np.column_stack((
        test_labels[:num_samples],
        anomaly_scores[:num_samples],
        gradient_weights
    ))

    # Write to CSV files
    with open(feature_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(feature_data.tolist())

    with open(gradient_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(gradient_data.tolist())


def find_optimal_ensemble_weight(
        scores_1: np.ndarray,
        scores_2: np.ndarray,
        labels: np.ndarray
) -> Tuple[float, float, float]:
    """
    Find the optimal weight (lambda) for ensembling two anomaly score models.

    Searches for the best linear combination: final_score = score_1 + lambda * score_2
    Both scores are standardized before combination.

    Args:
        scores_1: Anomaly scores from the first model
        scores_2: Anomaly scores from the second model
        labels: Ground truth labels

    Returns:
        Tuple containing:
            - best_auc: Best ROC AUC achieved
            - best_pr: Best Average Precision achieved
            - best_lambda: Optimal lambda weight for score_2
    """
    # Standardize scores
    scaler_1 = StandardScaler()
    scaler_2 = StandardScaler()

    scores_1_scaled = scaler_1.fit_transform(scores_1.reshape(-1, 1)).flatten()
    scores_2_scaled = scaler_2.fit_transform(scores_2.reshape(-1, 1)).flatten()

    # Define search space for lambda
    lambda_candidates = np.concatenate([
        np.arange(0, 1, 0.1),
        np.arange(1, 10, 1)
    ])

    best_auc = 0.0
    best_pr = 0.0
    best_lambda = 0.0

    # Grid search for optimal lambda
    for lambda_weight in lambda_candidates:
        combined_scores = scores_1_scaled + lambda_weight * scores_2_scaled
        auc, pr = calculate_auc_metrics(combined_scores, labels)

        # Use sum of AUC and PR as the optimization objective
        if auc + pr > best_auc + best_pr:
            best_auc = auc
            best_pr = pr
            best_lambda = lambda_weight

    return best_auc, best_pr, best_lambda


def setup_logger(
        log_file: str,
        verbosity: int = 1,
        logger_name: Optional[str] = None
) -> logging.Logger:
    """
    Create and configure a logger for training/evaluation.

    Args:
        log_file: Path to the log file
        verbosity: Logging level (0: DEBUG, 1: INFO, 2: WARNING)
        logger_name: Optional name for the logger

    Returns:
        Configured logger instance
    """
    level_mapping = {
        0: logging.DEBUG,
        1: logging.INFO,
        2: logging.WARNING
    }

    formatter = logging.Formatter("%(message)s")

    logger = logging.getLogger(logger_name)
    logger.setLevel(level_mapping.get(verbosity, logging.INFO))

    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    try:
        console_handler.stream.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    logger.addHandler(console_handler)

    return logger

def infer_input_dim(loader, fallback=None):
    """Dataloader veya dataset'ten güvenli şekilde input_dim çıkarır; başarısızsa fallback döner."""
    try:
        batch = next(iter(loader))
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return int(x.shape[1])
    except Exception:
        ds = getattr(loader, "dataset", None)
        if ds is not None:
            if hasattr(ds, "tensors") and len(ds.tensors) > 0:
                return int(ds.tensors[0].shape[1])
            if hasattr(ds, "data"):
                return int(ds.data.shape[1])
    if fallback is not None:
        return int(fallback)
    raise RuntimeError("input_dim çıkarılamadı: dataloader/dataset kontrol edin")
