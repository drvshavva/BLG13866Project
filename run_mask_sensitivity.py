import os
import argparse
from datetime import datetime
import warnings

import torch
import pandas as pd

from model.mcm import MCM
from model.trainer import MCMTrainer
from model.utils import initialize_weights
from data.loader import get_dataloader
from data.conf import DATASET_CONFIGS

warnings.filterwarnings('ignore')

def infer_input_dim(loader, fallback=None):
    try:
        batch = next(iter(loader))
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return int(x.shape[1])
    except Exception:
        ds = getattr(loader, 'dataset', None)
        if ds is not None:
            if hasattr(ds, 'tensors') and len(ds.tensors) > 0:
                return int(ds.tensors[0].shape[1])
            if hasattr(ds, 'data'):
                return int(ds.data.shape[1])
    if fallback is not None:
        return int(fallback)
    raise RuntimeError('Could not infer input_dim from loader or dataset')


def extract_xy_from_loader(loader):
    ds = getattr(loader, 'dataset', None)
    # common attributes
    if ds is not None:
        if hasattr(ds, 'tensors') and len(ds.tensors) >= 2:
            X = ds.tensors[0].cpu().numpy()
            y = ds.tensors[1].cpu().numpy()
            return X, y
        if hasattr(ds, 'data') and hasattr(ds, 'targets'):
            return ds.data, ds.targets

    # fallback: iterate through loader
    Xs, ys = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            bx = batch[0]
            by = batch[1] if len(batch) > 1 else None
        else:
            bx = batch
            by = None
        Xs.append(bx.cpu().numpy())
        if by is not None:
            ys.append(by.cpu().numpy())
    X = None
    y = None
    if len(Xs) > 0:
        X = np.concatenate(Xs, axis=0)
    if len(ys) > 0:
        y = np.concatenate(ys, axis=0)
    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask count sensitivity experiment for MCM on one dataset')
    parser.add_argument('--dataset', type=str, default='glass', help='dataset name as in DATASET_CONFIGS')
    parser.add_argument('--epochs', type=int, default=80, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='override batch size')
    parser.add_argument('--out', type=str, default='./results', help='output directory')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    dataset = args.dataset.lower()
    config = {
        'dataset_name': dataset,
        'data_dir': './datasets'
    }
    # load dataset specific config if present
    config.update(DATASET_CONFIGS.get(dataset, {}))
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    # dataloaders
    print(f'Loading dataset: {dataset.upper()}')
    train_loader, test_loader = get_dataloader(config)

    # infer input dim
    try:
        input_dim = infer_input_dim(train_loader, fallback=config.get('data_dim'))
    except Exception as e:
        raise RuntimeError(f'Failed to infer input dim: {e}')

    print(f'Detected input_dim = {input_dim}')

    # extract test arrays
    # lazy import numpy only if needed
    import numpy as np
    X_test_np, y_test_np = extract_xy_from_loader(test_loader)
    if X_test_np is None or y_test_np is None:
        raise RuntimeError('Could not extract X or y from test_loader; make sure your dataset provides labels')

    X_test_t = torch.FloatTensor(X_test_np)
    y_test_t = torch.FloatTensor(y_test_np)

    # experiment
    print('Deney 3: Maske Sayısı Duyarlılığı')
    print('=' * 50)

    mask_counts = [1, 3, 5, 10, 15, 20]
    results_masks = []

    for K in mask_counts:
        print(f'Running K={K}')
        model = MCM(input_dim=input_dim, hidden_dim=config.get('hidden_dim', 128), latent_dim=config.get('z_dim', 64), num_masks=K, lambda_div=config.get('lambda_div', 20.0))
        initialize_weights(model)
        trainer = MCMTrainer(model, learning_rate=config.get('learning_rate', 1e-3), device=device)

        trainer.train(train_loader, num_epochs=args.epochs, verbose=False)

        try:
            auc_roc, auc_pr = trainer.evaluate(X_test_t, y_test_t)
        except Exception as e:
            print(f'  Evaluation failed for K={K}: {e}')
            auc_roc, auc_pr = 0.5, 0.5

        results_masks.append({'K': K, 'AUC-ROC': float(auc_roc), 'AUC-PR': float(auc_pr)})
        print(f'K={K:2d}: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}\n')

    df_masks = pd.DataFrame(results_masks)

    os.makedirs(args.out, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(args.out, f'mask_sensitivity_{dataset}_{timestamp}.csv')
    df_masks.to_csv(csv_path, index=False)
    try:
        df_masks.to_excel(os.path.join(args.out, f'mask_sensitivity_{dataset}_{timestamp}.xlsx'), index=False)
    except Exception:
        pass

    print('Results saved to:', csv_path)
    print(df_masks.to_string(index=False))

