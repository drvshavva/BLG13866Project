import torch
from torch.utils.data import DataLoader, TensorDataset

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
    print("Exp 1: Detecting Different Types of Anomalies")
    print("=" * 50)

    anomaly_types = ['global', 'local', 'dependency', 'clustered']
    results_anomaly_types = []

    for atype in anomaly_types:
        print(f"\n{atype.upper()}:")

        X, y = generate_synthetic_data(
            n_normal=1500,
            n_anomaly=75,
            n_features=15,
            anomaly_type=atype
        )
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        split = split_data(X, y)

        # Model
        model = MCM(input_dim=15, hidden_dim=128, latent_dim=64, num_masks=10)
        initialize_weights(model)

        trainer = MCMTrainer(model=model, learning_rate=1e-3, device=device)

        train_dataset = TensorDataset(torch.FloatTensor(split['X_train']))
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        # Train
        trainer.train(train_loader, num_epochs=80, verbose=False)

        # Evaluate
        auc_roc, auc_pr = trainer.evaluate(
            torch.FloatTensor(split['X_test']),
            torch.FloatTensor(split['y_test'])
        )

        results_anomaly_types.append({
            'type': atype,
            'AUC-ROC': auc_roc,
            'AUC-PR': auc_pr
        })

        print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  AUC-PR: {auc_pr:.4f}")

    # Summary
    df_anomaly = pd.DataFrame(results_anomaly_types)
    print("\n" + "=" * 50)
    print(df_anomaly.to_string(index=False))
