import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.data_loader import get_dataloader
from .model import MaskedCellModelling
from .loss import AnomalyDetectionLoss
from .score import AnomalyScorer
from utils import calculate_auc_metrics, calculate_f1_score, setup_logger


class ModelTrainer:
    """
    Comprehensive trainer for anomaly detection models.

    Handles training, validation, evaluation, and checkpoint management
    for multi-masked contrastive models.

    Attributes:
        model: The neural network model
        loss_fn: Loss function module
        scorer: Anomaly scoring module
        optimizer: Optimization algorithm
        scheduler: Learning rate scheduler
        device: Device for computation (CPU/GPU)
        train_loader: Training data loader
        test_loader: Test data loader
        logger: Logger for training information
    """

    def __init__(
            self,
            model_config: Dict,
            run_id: int = 0,
            checkpoint_dir: Optional[str] = None,
            log_dir: Optional[str] = None
    ):
        """
        Initialize the trainer.

        Args:
            model_config: Configuration dictionary containing:
                - device (str): Device to use ('cuda' or 'cpu')
                - learning_rate (float): Initial learning rate
                - sche_gamma (float): Learning rate decay factor
                - weight_decay (float, optional): L2 regularization
                - optimizer (str, optional): Optimizer type ('adam', 'adamw', 'sgd')
                - scheduler_type (str, optional): Scheduler type ('exponential', 'step', 'cosine')
                And all model-specific parameters
            run_id: Identifier for this training run
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory to save training logs
        """
        # Store configuration FIRST
        self.run_id = run_id
        self.config = model_config

        # Set up directories SECOND
        self.checkpoint_dir = Path(checkpoint_dir or './checkpoints')
        self.log_dir = Path(log_dir or './logs')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging THIRD (before anything else that might log)
        log_file = self.log_dir / f'run_{run_id}_train.log'
        self.logger = setup_logger(str(log_file), verbosity=1, logger_name=f'trainer_{run_id}')

        self.logger.info("=" * 70)
        self.logger.info(f"Initializing Trainer for run {run_id}")
        self.logger.info("=" * 70)

        # NOW we can call methods that use self.logger
        # Device configuration
        self.logger.info("Setting up device...")
        self.device = self._setup_device(model_config.get('device', 'cuda'))
        model_config['device'] = str(self.device)

        # Initialize model, loss, and scorer
        self.logger.info("Building model components...")
        try:
            self.model = MaskedCellModelling(model_config).to(self.device)
            self.loss_fn = AnomalyDetectionLoss(model_config).to(self.device)
            self.scorer = AnomalyScorer(model_config).to(self.device)
            self.logger.info("Model components created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create model components: {e}")
            raise

        # Initialize optimizer
        self.logger.info("Setting up optimizer...")
        try:
            self.optimizer = self._setup_optimizer(model_config)
            self.logger.info("Optimizer created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create optimizer: {e}")
            raise

        # Initialize scheduler
        self.logger.info("Setting up learning rate scheduler...")
        try:
            self.scheduler = self._setup_scheduler(model_config)
            self.logger.info("Scheduler created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create scheduler: {e}")
            raise

        # Load data
        self.logger.info("Loading datasets...")
        try:
            self.train_loader, self.test_loader = get_dataloader(model_config)
            self.logger.info(f" Datasets loaded successfully")
            self.logger.info(f"  Training samples: {len(self.train_loader.dataset)}")
            self.logger.info(f"  Test samples: {len(self.test_loader.dataset)}")
        except Exception as e:
            self.logger.error(f" Failed to load datasets: {e}")
            raise

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_auc = 0.0
        self.training_history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_div_loss': [],
            'val_auc': [],
            'val_ap': [],
            'val_f1': [],
            'learning_rate': []
        }

        # Final summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info("=" * 70)
        self.logger.info("Trainer Initialization Complete")
        self.logger.info("=" * 70)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info("=" * 70)

    def _setup_device(self, device_str: str) -> torch.device:
        """Setup and validate device."""
        if device_str == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available. Using CPU.")
            device_str = 'cpu'

        device = torch.device(device_str)

        if device.type == 'cuda':
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.logger.info("Using CPU")

        return device

    def _setup_optimizer(self, config: Dict) -> optim.Optimizer:
        """Initialize optimizer based on configuration."""
        lr = config.get('learning_rate', 1e-3)
        weight_decay = config.get('weight_decay', 1e-5)
        optimizer_type = config.get('optimizer', 'adam').lower()

        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            momentum = config.get('momentum', 0.9)
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        self.logger.info(f"Optimizer: {optimizer_type}, LR: {lr}, Weight Decay: {weight_decay}")
        return optimizer

    def _setup_scheduler(self, config: Dict) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Initialize learning rate scheduler based on configuration."""
        scheduler_type = config.get('scheduler_type', 'exponential').lower()

        if scheduler_type == 'none':
            return None

        if scheduler_type == 'exponential':
            gamma = config.get('sche_gamma', 0.95)
            scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
            self.logger.info(f"Scheduler: ExponentialLR, gamma: {gamma}")

        elif scheduler_type == 'step':
            step_size = config.get('step_size', 10)
            gamma = config.get('sche_gamma', 0.5)
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
            self.logger.info(f"Scheduler: StepLR, step_size: {step_size}, gamma: {gamma}")

        elif scheduler_type == 'cosine':
            T_max = config.get('T_max', 50)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
            self.logger.info(f"Scheduler: CosineAnnealingLR, T_max: {T_max}")

        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.get('sche_gamma', 0.5),
                patience=config.get('patience', 5),
                verbose=True
            )
            self.logger.info("Scheduler: ReduceLROnPlateau")

        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler

    def train_epoch(self) -> Tuple[float, float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, average_recon_loss, average_div_loss)
        """
        self.model.train()

        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_div_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch}',
            leave=False
        )

        for x_input, _ in pbar:
            x_input = x_input.to(self.device)

            # Forward pass
            x_pred, z, masks = self.model(x_input)

            # Compute loss
            loss, recon_loss, div_loss = self.loss_fn(x_input, x_pred, masks)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Optional gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()

            # Accumulate losses
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_div_loss += div_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'div': f'{div_loss.item():.4f}'
            })

        # Average losses
        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_div_loss = epoch_div_loss / num_batches

        return avg_loss, avg_recon_loss, avg_div_loss

    def validate(self) -> Tuple[float, float, float]:
        """
        Validate on test set.

        Returns:
            Tuple of (roc_auc, average_precision, f1_score)
        """
        self.model.eval()

        all_scores = []
        all_labels = []

        with torch.no_grad():
            for x_input, y_label in tqdm(self.test_loader, desc='Validating', leave=False):
                x_input = x_input.to(self.device)

                # Forward pass
                x_pred, _, _ = self.model(x_input)

                # Compute anomaly scores
                scores = self.scorer(x_input, x_pred)

                all_scores.append(scores.cpu())
                all_labels.append(y_label)

        # Concatenate results
        all_scores = torch.cat(all_scores, dim=0).numpy().flatten()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # Compute metrics
        roc_auc, avg_precision = calculate_auc_metrics(all_scores, all_labels)
        f1 = calculate_f1_score(all_scores, all_labels)

        return roc_auc, avg_precision, f1

    def train(
            self,
            num_epochs: int,
            validate_every: int = 1,
            save_best: bool = True,
            early_stopping: bool = True,
            patience: int = 10,
            save_last: bool = True
    ) -> Dict:
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
            validate_every: Validate every N epochs
            save_best: Whether to save best model based on validation AUC
            early_stopping: Whether to use early stopping
            patience: Number of epochs without improvement before stopping
            save_last: Whether to save the last model checkpoint

        Returns:
            Dictionary containing training history

        Example:
            >>> trainer = ModelTrainer(config)
            >>> history = trainer.train(num_epochs=100, patience=10)
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")

        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Training phase
            train_loss, train_recon, train_div = self.train_epoch()

            # Learning rate step (for non-plateau schedulers)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log training metrics
            log_msg = (
                f"Epoch [{epoch}/{num_epochs}] "
                f"Loss: {train_loss:.4f} "
                f"(Recon: {train_recon:.4f}, Div: {train_div:.4f}) "
                f"LR: {current_lr:.6f}"
            )

            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_recon_loss'].append(train_recon)
            self.training_history['train_div_loss'].append(train_div)
            self.training_history['learning_rate'].append(current_lr)

            # Validation phase
            if (epoch + 1) % validate_every == 0:
                roc_auc, avg_precision, f1 = self.validate()

                log_msg += f" | Val - AUC: {roc_auc:.4f}, AP: {avg_precision:.4f}, F1: {f1:.4f}"

                # Store validation metrics
                self.training_history['val_auc'].append(roc_auc)
                self.training_history['val_ap'].append(avg_precision)
                self.training_history['val_f1'].append(f1)

                # Update scheduler if plateau
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_loss)

                # Save best model
                if save_best and roc_auc > self.best_auc:
                    self.best_auc = roc_auc
                    self.best_loss = train_loss
                    self.save_checkpoint('best_model.pth', is_best=True)
                    log_msg += " [Best Model Saved]"
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                # Early stopping check
                if early_stopping and epochs_without_improvement >= patience:
                    self.logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs "
                        f"(no improvement for {patience} epochs)"
                    )
                    break
            else:
                # Update scheduler for non-validation epochs
                if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()

            self.logger.info(log_msg)

        # Save final model
        if save_last:
            self.save_checkpoint('last_model.pth', is_best=False)

        self.logger.info("Training completed!")
        self.logger.info(f"Best validation AUC: {self.best_auc:.4f}")

        return self.training_history

    def evaluate(
            self,
            checkpoint_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation on test set.

        Args:
            checkpoint_path: Path to model checkpoint. If None, uses current model.

        Returns:
            Dictionary containing all evaluation metrics

        Example:
            >>> trainer = ModelTrainer(config)
            >>> metrics = trainer.evaluate('best_model.pth')
            >>> print(f"AUC-ROC: {metrics['roc_auc']:.4f}")
        """
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            self.logger.info(f"Loaded checkpoint: {checkpoint_path}")

        self.model.eval()

        all_scores = []
        all_labels = []
        all_features = []

        with torch.no_grad():
            for x_input, y_label in tqdm(self.test_loader, desc='Evaluating', leave=False):
                x_input = x_input.to(self.device)

                # Forward pass
                x_pred, latent, masks = self.model(x_input)

                # Compute anomaly scores
                scores = self.scorer(x_input, x_pred)

                all_scores.append(scores.cpu())
                all_labels.append(y_label)
                all_features.append(x_input.cpu())

        # Concatenate results
        all_scores = torch.cat(all_scores, dim=0).numpy().flatten()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # Compute metrics
        roc_auc, avg_precision = calculate_auc_metrics(all_scores, all_labels)
        f1 = calculate_f1_score(all_scores, all_labels)

        metrics = {
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'f1_score': f1,
            'num_samples': len(all_labels),
            'num_anomalies': int(all_labels.sum()),
            'anomaly_rate': float(all_labels.mean())
        }

        # Log results
        self.logger.info("=" * 70)
        self.logger.info("Evaluation Results:")
        self.logger.info("=" * 70)
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 70)

        return metrics

    def save_checkpoint(
            self,
            filename: str,
            is_best: bool = False
    ):
        """
        Save model checkpoint.

        Args:
            filename: Name of checkpoint file
            is_best: Whether this is the best model so far
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'best_auc': self.best_auc,
            'config': self.config,
            'training_history': self.training_history
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Also save full model for backward compatibility
        # Use _use_new_zipfile_serialization=False for compatibility
        if is_best:
            model_path = self.checkpoint_dir / 'model.pth'
            try:
                torch.save(self.model, model_path, _use_new_zipfile_serialization=False)
            except TypeError:
                # Older PyTorch version
                torch.save(self.model, model_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            # Try in checkpoint directory
            checkpoint_path = self.checkpoint_dir / checkpoint_path.name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False  # Required for PyTorch 2.6+
            )
        except TypeError:
            # Fallback for older PyTorch versions that don't have weights_only parameter
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.best_auc = checkpoint.get('best_auc', 0.0)
            self.training_history = checkpoint.get('training_history', self.training_history)
        else:
            # Old format: full model
            self.model = checkpoint

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


# Backward compatibility wrapper
class Trainer:
    """Legacy trainer wrapper for backward compatibility."""

    def __init__(self, run: int, model_config: dict):
        """Initialize legacy trainer."""
        self.trainer = ModelTrainer(model_config, run_id=run)
        self.run = run

        # Store metrics arrays (for legacy interface)
        self.mse_rauc = np.zeros(1)
        self.mse_ap = np.zeros(1)
        self.mse_f1 = np.zeros(1)

    def training(self, epochs: int):
        """Train the model (legacy interface)."""
        self.trainer.train(
            num_epochs=epochs,
            validate_every=epochs // 10 if epochs > 10 else 1,
            early_stopping=False
        )

    def evaluate(self, mse_rauc, mse_ap, mse_f1):
        """Evaluate the model (legacy interface)."""
        metrics = self.trainer.evaluate(
            checkpoint_path=str(self.trainer.checkpoint_dir / 'best_model.pth')
        )

        mse_rauc[self.run] = metrics['roc_auc']
        mse_ap[self.run] = metrics['average_precision']
        mse_f1[self.run] = metrics['f1_score']


