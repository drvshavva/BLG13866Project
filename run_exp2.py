import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

import pandas as pd
import warnings

from model.mcm import MCM

from model.trainer import MCMTrainer
from data.syn_data import generate_synthetic_data
from data.split import split_data
from model.utils import initialize_weights

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

if __name__ == "__main__":
    print("Exp 2: Ablation Study on MCM Components")
    print("=" * 50)

    # Base data
    X, y = generate_synthetic_data(n_normal=1500, n_anomaly=75, n_features=15)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    split = split_data(X, y)

    X_train_t = torch.FloatTensor(split['X_train'])
    X_test_t = torch.FloatTensor(split['X_test'])
    y_test_t = torch.FloatTensor(split['y_test'])

    train_dataset = TensorDataset(X_train_t)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    ablation_results = []

    # Task A: Vanilla AE (no masking)
    print("\nTask A: Vanilla Autoencoder")


    class VanillaAE(nn.Module):
        def __init__(self, input_dim, hidden_dim=256, latent_dim=128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, latent_dim), nn.LeakyReLU(0.2)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, input_dim)
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))


    ae_model = VanillaAE(15, 128, 64).to(device)
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=1e-3)

    for _ in range(80):
        for batch in train_loader:
            x = batch[0].to(device)
            x_hat = ae_model(x)
            loss = F.mse_loss(x_hat, x)
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()

    ae_model.eval()
    with torch.no_grad():
        x_hat = ae_model(X_test_t.to(device))
        scores = torch.sum((x_hat - X_test_t.to(device)) ** 2, dim=1).cpu().numpy()

    auc_roc = roc_auc_score(split['y_test'], scores)
    auc_pr = average_precision_score(split['y_test'], scores)
    ablation_results.append({'Task': 'A: Vanilla AE', 'AUC-ROC': auc_roc, 'AUC-PR': auc_pr})
    print(f"  AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")

    # Task C: Single Mask
    print("\nTask C: Single Mask (K=1)")
    model_c = MCM(input_dim=15, hidden_dim=128, latent_dim=64, num_masks=1)
    initialize_weights(model_c)
    trainer_c = MCMTrainer(model_c, learning_rate=1e-3, device=device)
    trainer_c.train(train_loader, num_epochs=80, verbose=False)
    auc_roc, auc_pr = trainer_c.evaluate(X_test_t, y_test_t)
    ablation_results.append({'Task': 'C: Single Mask', 'AUC-ROC': auc_roc, 'AUC-PR': auc_pr})
    print(f"  AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")

    # Task D: No Diversity Loss
    print("\nTask D: No Diversity Loss")
    model_d = MCM(input_dim=15, hidden_dim=128, latent_dim=64, num_masks=10, lambda_div=0)
    initialize_weights(model_d)
    trainer_d = MCMTrainer(model_d, learning_rate=1e-3, device=device)
    trainer_d.train(train_loader, num_epochs=80, verbose=False)
    auc_roc, auc_pr = trainer_d.evaluate(X_test_t, y_test_t)
    ablation_results.append({'Task': 'D: No Diversity', 'AUC-ROC': auc_roc, 'AUC-PR': auc_pr})
    print(f"  AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")

    # Task E: Full MCM
    print("\nTask E: Full MCM")
    model_e = MCM(input_dim=15, hidden_dim=128, latent_dim=64, num_masks=10, lambda_div=10)
    initialize_weights(model_e)
    trainer_e = MCMTrainer(model_e, learning_rate=1e-3, device=device)
    trainer_e.train(train_loader, num_epochs=80, verbose=False)
    auc_roc, auc_pr = trainer_e.evaluate(X_test_t, y_test_t)
    ablation_results.append({'Task': 'E: Full MCM', 'AUC-ROC': auc_roc, 'AUC-PR': auc_pr})
    print(f"  AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")

    # Summary
    df_ablation = pd.DataFrame(ablation_results)
    print("\n" + "=" * 50)
    print(df_ablation.to_string(index=False))