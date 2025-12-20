import os

import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

import numpy as np
import pandas as pd
import warnings

from model.mcm import MCM
from model.trainer import MCMTrainer
from model.utils import initialize_weights
from data.loader import get_dataloader
from data.conf import DATASET_CATEGORIES, DATASET_CONFIGS

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def extract_xy_from_loader(loader):
    dataset = loader.dataset
    # common patterns
    if hasattr(dataset, "tensors"):
        X = dataset.tensors[0].cpu().numpy()
        y = dataset.tensors[1].cpu().numpy() if len(dataset.tensors) > 1 else None
        return X, y
    if hasattr(dataset, "data") and hasattr(dataset, "targets"):
        return dataset.data, dataset.targets
    # fallback: iterate loader and collect
    Xs, ys = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            bx = batch[0]
            by = batch[1] if len(batch) > 1 else None
        else:
            bx, by = batch, None
        Xs.append(bx.cpu().numpy())
        if by is not None:
            ys.append(by.cpu().numpy())
    X = np.concatenate(Xs, axis=0) if Xs else None
    y = np.concatenate(ys, axis=0) if ys else None
    return X, y


# Vanilla AE model
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


if __name__ == "__main__":
    results = []

    # common base config
    base_config = {
        'data_dir': './datasets',
        'num_workers': 4
    }

    for dataset in DATASET_CATEGORIES['all']:
        print(f"\nRunning dataset: {dataset}")

        # For each dataset, create a fresh config
        config = base_config.copy()
        config['dataset_name'] = dataset
        config.update(DATASET_CONFIGS.get(dataset, {}))

        # take dataloader
        train_loader, test_loader = get_dataloader(config)

        X_test_np, y_test_np = extract_xy_from_loader(test_loader)
        # convert to tensors
        X_test_t = torch.FloatTensor(X_test_np).to(device)
        y_test_t = torch.FloatTensor(y_test_np).to(device) if y_test_np is not None else None

        ablation_results = []

        # Task A: Vanilla AE
        input_dim = config.get('data_dim', X_test_np.shape[1])
        print(" Task A: Vanilla AE")
        ae = VanillaAE(input_dim, hidden_dim=128, latent_dim=64).to(device)
        ae_opt = optim.Adam(ae.parameters(), lr=1e-3)
        ae.train()
        for _ in range(80):
            for batch in train_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device)
                x_hat = ae(x)
                loss = F.mse_loss(x_hat, x)
                ae_opt.zero_grad()
                loss.backward()
                ae_opt.step()
        ae.eval()
        with torch.no_grad():
            x_hat = ae(X_test_t)
            scores = torch.sum((x_hat - X_test_t) ** 2, dim=1).cpu().numpy()
        auc_roc = roc_auc_score(y_test_np, scores)
        auc_pr = average_precision_score(y_test_np, scores)
        ablation_results.append({'Task': 'A: Vanilla AE', 'AUC-ROC': float(auc_roc), 'AUC-PR': float(auc_pr)})
        print(f"  AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")

        # Task C: Single Mask
        print(" Task C: Single Mask (K=1)")
        model_c = MCM(input_dim=input_dim, hidden_dim=128, latent_dim=64, num_masks=1)
        initialize_weights(model_c)
        model_c.to(device)
        trainer_c = MCMTrainer(model_c, learning_rate=1e-3, device=device)
        trainer_c.train(train_loader, num_epochs=80, verbose=False)
        auc_roc, auc_pr = trainer_c.evaluate(X_test_t, y_test_np)
        ablation_results.append({'Task': 'C: Single Mask', 'AUC-ROC': float(auc_roc), 'AUC-PR': float(auc_pr)})
        print(f"  AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")

        # Task D: No Diversity Loss
        print(" Task D: No Diversity Loss")
        model_d = MCM(input_dim=input_dim, hidden_dim=128, latent_dim=64, num_masks=10, lambda_div=0)
        initialize_weights(model_d)
        model_d.to(device)
        trainer_d = MCMTrainer(model_d, learning_rate=1e-3, device=device)
        trainer_d.train(train_loader, num_epochs=80, verbose=False)
        auc_roc, auc_pr = trainer_d.evaluate(X_test_t, y_test_np)
        ablation_results.append({'Task': 'D: No Diversity', 'AUC-ROC': float(auc_roc), 'AUC-PR': float(auc_pr)})
        print(f"  AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")

        # Task E: Full MCM
        print(" Task E: Full MCM")
        model_e = MCM(input_dim=input_dim, hidden_dim=128, latent_dim=64, num_masks=10, lambda_div=10)
        initialize_weights(model_e)
        model_e.to(device)
        trainer_e = MCMTrainer(model_e, learning_rate=1e-3, device=device)
        trainer_e.train(train_loader, num_epochs=80, verbose=False)
        auc_roc, auc_pr = trainer_e.evaluate(X_test_t, y_test_np)
        ablation_results.append({'Task': 'E: Full MCM', 'AUC-ROC': float(auc_roc), 'AUC-PR': float(auc_pr)})
        print(f"  AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")

        for r in ablation_results:
            r_record = {
                'dataset': dataset,
                'task': r['Task'],
                'AUC-ROC': r['AUC-ROC'],
                'AUC-PR': r['AUC-PR'],
                'timestamp': datetime.datetime.now().isoformat()
            }
            results.append(r_record)

    df = pd.DataFrame(results)
    # compute mean per task across datasets
    df_mean = df.groupby('task')[['AUC-ROC', 'AUC-PR']].mean().reset_index()
    out_dir = './results'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'ablation_per_dataset.csv')
    csv_mean = os.path.join(out_dir, 'ablation_mean_across_datasets.csv')
    df.to_csv(csv_path, index=False)
    df_mean.to_csv(csv_mean, index=False)
    try:
        df.to_excel(os.path.join(out_dir, 'ablation_per_dataset.xlsx'), index=False)
        df_mean.to_excel(os.path.join(out_dir, 'ablation_mean_across_datasets.xlsx'), index=False)
    except Exception:
        pass

    print("\nPer-dataset results saved to:", csv_path)
    print("Averaged results saved to:", csv_mean)
    print("\nAveraged results:")
    print(df_mean.to_string(index=False))
