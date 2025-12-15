import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

from typing import Dict, List, Tuple, Optional
from tqdm.notebook import tqdm

from .mcm import MCM

class MCMTrainer:
    """
    - Epochs: 200
    - Batch size: 512
    - Optimizer: Adam with exponential decay
    """

    def __init__(
        self,
        model: MCM,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        lr_decay: float = 0.99,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=lr_decay
        )

        self.history = {
            'train_loss': [],
            'rec_loss': [],
            'div_loss': [],
            'val_auc_roc': [],
            'val_auc_pr': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """One epoch of training"""
        self.model.train()

        total_loss = 0.0
        total_rec_loss = 0.0
        total_div_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            if isinstance(batch_data, (list, tuple)):
                x = batch_data[0].to(self.device)
            else:
                x = batch_data.to(self.device)

            x_reconstructed, masks, _ = self.model(x)
            loss, rec_loss, div_loss = self.model.compute_total_loss(x, x_reconstructed, masks)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            if type(div_loss) is float:
                total_div_loss += div_loss
            else:
                total_div_loss += div_loss.item()
            num_batches += 1

        self.scheduler.step()

        return {
            'loss': total_loss / num_batches,
            'rec_loss': total_rec_loss / num_batches,
            'div_loss': total_div_loss / num_batches
        }

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 200,
        val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Model training"""
        for epoch in range(num_epochs):
            metrics = self.train_epoch(train_loader)

            self.history['train_loss'].append(metrics['loss'])
            self.history['rec_loss'].append(metrics['rec_loss'])
            self.history['div_loss'].append(metrics['div_loss'])

            if val_data is not None:
                X_val, y_val = val_data
                auc_roc, auc_pr = self.evaluate(X_val, y_val)
                self.history['val_auc_roc'].append(auc_roc)
                self.history['val_auc_pr'].append(auc_pr)

        return self.history

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """Model evaluation"""
        self.model.eval()

        X = X.to(self.device)

        with torch.no_grad():
            scores = self.model.compute_anomaly_score(X)

        scores = scores.cpu().numpy()
        y = y.numpy() if isinstance(y, torch.Tensor) else y

        try:
            auc_roc = roc_auc_score(y, scores)
            auc_pr = average_precision_score(y, scores)
        except ValueError:
            auc_roc = 0.5
            auc_pr = 0.5

        return auc_roc, auc_pr

    def predict_anomaly_scores(self, X: torch.Tensor) -> np.ndarray:
        """Predict anomaly scores for given data"""
        self.model.eval()
        X = X.to(self.device)

        with torch.no_grad():
            scores = self.model.compute_anomaly_score(X)

        return scores.cpu().numpy()