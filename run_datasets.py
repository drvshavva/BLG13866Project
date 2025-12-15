import os
from datetime import datetime

import torch
import pandas as pd
import warnings

from model.mcm import MCM

from model.trainer import MCMTrainer
from model.utils import initialize_weights
from data.loader import get_dataloader

from utils import infer_input_dim

warnings.filterwarnings('ignore')
"""
Running dataset: THYROID
Loaded train dataset with 1839 samples
Loaded test dataset with 1933 samples
  AUC-ROC: 0.8400
  AUC-PR: 0.2482
Running dataset: MAMMOGRAPHY
Loaded train dataset with 5461 samples
Loaded test dataset with 5722 samples
  AUC-ROC: 0.8874
  AUC-PR: 0.3282
Running dataset: PIMA
Loaded train dataset with 250 samples
Loaded test dataset with 518 samples
  AUC-ROC: 0.6946
  AUC-PR: 0.6895
Running dataset: BREASTW
Loaded train dataset with 222 samples
Loaded test dataset with 461 samples
  AUC-ROC: 0.9971
  AUC-PR: 0.9972
Running dataset: SHUTTLE
Loaded train dataset with 22793 samples
Loaded test dataset with 26304 samples
  AUC-ROC: 0.9960
  AUC-PR: 0.9258
Running dataset: GLASS
Loaded train dataset with 102 samples
Loaded test dataset with 112 samples
  AUC-ROC: 0.7184
  AUC-PR: 0.1977
Running dataset: WINE
Loaded train dataset with 59 samples
Loaded test dataset with 70 samples
  AUC-ROC: 0.5000
  AUC-PR: 0.5000
Running dataset: PENDIGITS
Loaded train dataset with 3357 samples
Loaded test dataset with 3513 samples
  AUC-ROC: 0.9425
  AUC-PR: 0.3331
Running dataset: CARDIO
Loaded train dataset with 827 samples
Loaded test dataset with 1004 samples
  AUC-ROC: 0.5000
  AUC-PR: 0.5000
Running dataset: CARDIOTOCOGRAPHY
Loaded train dataset with 824 samples
Loaded test dataset with 1290 samples
  AUC-ROC: 0.5000
  AUC-PR: 0.5000
Running dataset: FRAUD
Loaded train dataset with 142157 samples
Loaded test dataset with 142650 samples
  AUC-ROC: 0.9248
  AUC-PR: 0.4923
Running dataset: WBC
Loaded train dataset with 178 samples
Loaded test dataset with 200 samples
  AUC-ROC: 0.8055
  AUC-PR: 0.3599
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET_CONFIGS = {
    'thyroid': {
        'data_dim': 6,
        'hidden_dim': 64,
        'z_dim': 32,
        'mask_num': 5,
        'learning_rate': 0.05,
        'batch_size': 256,
    },

    'mammography': {
        'data_dim': 6,
        'hidden_dim': 64,
        'z_dim': 32,
        'mask_num': 5,
        'learning_rate': 0.05,
        'batch_size': 512,
    },

    'pima': {
        'data_dim': 8,
        'hidden_dim': 64,
        'z_dim': 32,
        'mask_num': 6,
        'learning_rate': 0.05,
        'batch_size': 256,
    },

    'breastw': {
        'data_dim': 9,
        'hidden_dim': 64,
        'z_dim': 32,
        'mask_num': 6,
        'learning_rate': 0.05,
        'batch_size': 256,
    },

    'shuttle': {
        'data_dim': 9,
        'hidden_dim': 64,
        'z_dim': 32,
        'mask_num': 6,
        'learning_rate': 0.05,
        'batch_size': 512,
    },

    'glass': {
        'data_dim': 7,
        'hidden_dim': 64,
        'z_dim': 32,
        'mask_num': 6,
        'learning_rate': 0.05,
        'batch_size': 128,
    },

    'wine': {
        'data_dim': 13,
        'hidden_dim': 96,
        'z_dim': 48,
        'mask_num': 8,
        'learning_rate': 0.05,
        'batch_size': 256,
    },

    'pendigits': {
        'data_dim': 16,
        'hidden_dim': 128,
        'z_dim': 64,
        'mask_num': 10,
        'learning_rate': 0.05,
        'batch_size': 512,
    },

    'cardio': {
        'data_dim': 21,
        'hidden_dim': 128,
        'z_dim': 64,
        'mask_num': 10,
        'learning_rate': 0.05,
        'batch_size': 512,
    },

    'cardiotocography': {
        'data_dim': 21,
        'hidden_dim': 128,
        'z_dim': 64,
        'mask_num': 10,
        'learning_rate': 0.05,
        'batch_size': 512,
    },

    'fraud': {
        'data_dim': 29,
        'hidden_dim': 128,
        'z_dim': 64,
        'mask_num': 12,
        'learning_rate': 0.05,
        'batch_size': 512,
    },

    'wbc': {
        'data_dim': 30,
        'hidden_dim': 256,
        'z_dim': 128,
        'mask_num': 15,
        'learning_rate': 0.05,
        'batch_size': 512,
    },

    'ionosphere': {
        'data_dim': 33,
        'hidden_dim': 128,
        'z_dim': 64,
        'mask_num': 10,
        'learning_rate': 0.05,
        'batch_size': 256,
    },

    'satellite': {
        'data_dim': 36,
        'hidden_dim': 128,
        'z_dim': 64,
        'mask_num': 12,
        'learning_rate': 0.05,
        'batch_size': 512,
    },

    'satimage-2': {
        'data_dim': 36,
        'hidden_dim': 128,
        'z_dim': 64,
        'mask_num': 12,
        'learning_rate': 0.05,
        'batch_size': 512,
    },

    'campaign': {
        'data_dim': 62,
        'hidden_dim': 256,
        'z_dim': 128,
        'mask_num': 20,
        'learning_rate': 0.03,
        'batch_size': 512,
    },

    'optdigits': {
        'data_dim': 64,
        'hidden_dim': 256,
        'z_dim': 128,
        'mask_num': 20,
        'learning_rate': 0.03,
        'batch_size': 512,
    },

    'nslkdd': {
        'data_dim': 122,
        'hidden_dim': 512,
        'z_dim': 256,
        'mask_num': 30,
        'learning_rate': 0.01,
        'batch_size': 512,
        'epochs': 150,
    },

    'arrhythmia': {
        'data_dim': 274,
        'hidden_dim': 512,
        'z_dim': 256,
        'mask_num': 40,
        'learning_rate': 0.01,
        'batch_size': 512,
        'epochs': 150,
    },

    'census': {
        'data_dim': 299,
        'hidden_dim': 512,
        'z_dim': 256,
        'mask_num': 50,
        'learning_rate': 0.01,
        'batch_size': 512,
        'epochs': 150,
    },
}

DATASET_CATEGORIES = {
    'medium': [ 'ionosphere', 'satellite', 'satimage-2', 'campaign',
               'optdigits'],
    'large': ['census'],
}

DATASET_CATEGORIES['all'] = (
        DATASET_CATEGORIES['medium'] +
        DATASET_CATEGORIES['large']
)



if __name__ == "__main__":
    config = {
        'dataset_name': 'thyroid',
        'data_dim': 6,
        'data_dir': './datasets',
        'batch_size': 256,
        'num_workers': 4}
    results = []

    for dataset in DATASET_CATEGORIES['all']:
        print(f"Running dataset: {dataset.upper()}")
        config['dataset_name'] = dataset
        config.update(DATASET_CONFIGS.get(dataset, {}))

        # data
        train_loader, test_loader = get_dataloader(config)
        input_dim = infer_input_dim(train_loader, fallback=config.get('data_dim'))
        print(f"Detected input_dim: {input_dim}")

        # model
        model = MCM(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            latent_dim=config['z_dim'],
            num_masks=config['mask_num']
        )
        initialize_weights(model)

        trainer = MCMTrainer(
            model=model,
            learning_rate=config.get('learning_rate', 1e-3),
            device=device
        )

        # Train
        trainer.train(train_loader, num_epochs=config.get('epochs', 50))

        # Evaluate
        auc_roc, auc_pr = trainer.evaluate(
            test_loader.dataset.data,
            test_loader.dataset.targets
        )

        results.append({
            'dataset': dataset,
            'AUC-ROC': float(auc_roc),
            'AUC-PR': float(auc_pr),
            'timestamp': datetime.now().isoformat(),
            'epochs': config.get('epochs', 50),
            'batch_size': config.get('batch_size')
        })

        print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  AUC-PR: {auc_pr:.4f}")

    out_dir = './results'
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)

    csv_path = os.path.join(out_dir, 'results_all_datasets.csv')
    xlsx_path = os.path.join(out_dir, 'results_all_datasets.xlsx')

    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception:
        pass

    print(f"Results saved to: {csv_path}")
    if os.path.exists(xlsx_path):
        print(f"Results also saved to: {xlsx_path}")