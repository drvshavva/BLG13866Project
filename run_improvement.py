import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

from model.mcm import MCM
from model.trainer import MCMTrainer
from model.utils import initialize_weights
from data.loader import get_dataloader
from data.conf import DATASET_CONFIGS

warnings.filterwarnings('ignore')

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print("=" * 70)


def extract_xy_from_loader(loader):
    """Extract X and y tensors from DataLoader."""
    dataset = loader.dataset

    # Try common dataset patterns
    if hasattr(dataset, "tensors"):
        X = dataset.tensors[0].cpu().numpy()
        y = dataset.tensors[1].cpu().numpy() if len(dataset.tensors) > 1 else None
        return X, y

    if hasattr(dataset, "data") and hasattr(dataset, "targets"):
        return dataset.data, dataset.targets

    # Fallback: iterate through loader
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


def infer_input_dim(loader, fallback=None):
    """Infer input dimension from DataLoader."""
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
    raise RuntimeError('Could not infer input_dim from loader')


def train_and_evaluate(model, model_name, train_loader, X_test, y_test,
                       num_epochs, device, verbose=True):
    """Train model and return evaluation metrics."""
    print(f"\n{'=' * 70}")
    print(f"Training: {model_name}")
    print(f"{'=' * 70}")

    # Initialize weights
    initialize_weights(model)
    model.to(device)

    # Create trainer
    trainer = MCMTrainer(model, learning_rate=1e-3, device=device)

    # Train
    print("Training in progress...")
    trainer.train(train_loader, num_epochs=num_epochs, verbose=verbose)

    # Evaluate
    print("Evaluating...")
    X_test_t = torch.FloatTensor(X_test).to(device)
    auc_roc, auc_pr = trainer.evaluate(X_test_t, y_test)

    print(f"\nResults for {model_name}:")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  AUC-PR:  {auc_pr:.4f}")

    return {
        'model_name': model_name,
        'AUC-ROC': float(auc_roc),
        'AUC-PR': float(auc_pr),
        'trainer': trainer
    }


def plot_comparison(results, dataset_name, output_dir='./results'):
    """Create comparison visualization."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = [r['model_name'] for r in results]
    auc_rocs = [r['AUC-ROC'] for r in results]
    auc_prs = [r['AUC-PR'] for r in results]

    colors = ['#3498db', '#e74c3c']

    # AUC-ROC comparison
    bars1 = axes[0].bar(models, auc_rocs, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('AUC-ROC Score', fontsize=12, fontweight='bold')
    axes[0].set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom', fontweight='bold')

    # AUC-PR comparison
    bars2 = axes[1].bar(models, auc_prs, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('AUC-PR Score', fontsize=12, fontweight='bold')
    axes[1].set_title('AUC-PR Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f'Baseline MCM vs Attention-Based MCM\nDataset: {dataset_name.upper()}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(output_dir, f'comparison_{dataset_name}_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_path}")

    plt.show()


def main():
    # Configuration
    dataset_name = 'thyroid'  # Change this to test other datasets
    num_epochs = 100
    num_masks = 10
    hidden_dim = 128
    latent_dim = 64

    print(f"COMPARISON EXPERIMENT")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Epochs: {num_epochs}")
    print(f"Number of Masks: {num_masks}")
    print("=" * 70)

    # Prepare dataset configuration
    config = {
        'dataset_name': dataset_name,
        'data_dir': './datasets',
        'batch_size': 512
    }
    config.update(DATASET_CONFIGS.get(dataset_name, {}))

    # Load dataset
    print(f"\nLoading dataset: {dataset_name.upper()}...")
    train_loader, test_loader = get_dataloader(config)

    # Infer input dimension
    try:
        input_dim = infer_input_dim(train_loader, fallback=config.get('data_dim'))
    except Exception as e:
        raise RuntimeError(f'Failed to infer input dim: {e}')

    print(f"Detected input_dim = {input_dim}")

    # Extract test data
    X_test_np, y_test_np = extract_xy_from_loader(test_loader)
    if X_test_np is None or y_test_np is None:
        raise RuntimeError('Could not extract X or y from test_loader')

    print(f"Train batches: {len(train_loader)}")
    print(f"Test samples: {len(X_test_np)}")
    print(f"Anomaly ratio: {y_test_np.mean():.2%}")

    # Store results
    all_results = []

    # ==========================================
    # MODEL 1: Baseline MCM (Static Masks)
    # ==========================================
    model_baseline = MCM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_masks=num_masks,
        lambda_div=10.0,
        use_attention_mask=False  # Baseline: Static masks
    )

    result_baseline = train_and_evaluate(
        model=model_baseline,
        model_name='Baseline MCM',
        train_loader=train_loader,
        X_test=X_test_np,
        y_test=y_test_np,
        num_epochs=num_epochs,
        device=device,
        verbose=False
    )
    all_results.append(result_baseline)

    # ==========================================
    # MODEL 2: Attention-Based MCM
    # ==========================================
    model_attention = MCM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_masks=num_masks,
        lambda_div=10.0,
        use_attention_mask=True,  # NEW: Attention-based masks
        n_heads=4  # Number of attention heads
    )

    result_attention = train_and_evaluate(
        model=model_attention,
        model_name='Attention-Based MCM',
        train_loader=train_loader,
        X_test=X_test_np,
        y_test=y_test_np,
        num_epochs=num_epochs,
        device=device,
        verbose=False
    )
    all_results.append(result_attention)

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    df_results = pd.DataFrame([
        {
            'Model': r['model_name'],
            'AUC-ROC': r['AUC-ROC'],
            'AUC-PR': r['AUC-PR']
        } for r in all_results
    ])

    print("\n" + df_results.to_string(index=False))

    # Calculate improvements
    baseline_roc = result_baseline['AUC-ROC']
    baseline_pr = result_baseline['AUC-PR']
    attention_roc = result_attention['AUC-ROC']
    attention_pr = result_attention['AUC-PR']

    improvement_roc = ((attention_roc - baseline_roc) / baseline_roc) * 100
    improvement_pr = ((attention_pr - baseline_pr) / baseline_pr) * 100

    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)
    print(f"AUC-ROC Improvement: {improvement_roc:+.2f}%")
    print(f"AUC-PR Improvement:  {improvement_pr:+.2f}%")

    if improvement_roc > 0 and improvement_pr > 0:
        print("\n✓ Attention-based method shows improvement on both metrics!")
    elif improvement_roc > 0 or improvement_pr > 0:
        print("\n⚠ Attention-based method shows mixed results.")
    else:
        print("\n✗ Baseline method performed better.")
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save to CSV
    csv_path = os.path.join(output_dir, f'comparison_{dataset_name}_{timestamp}.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save detailed report
    report_path = os.path.join(output_dir, f'comparison_report_{dataset_name}_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("COMPARISON REPORT: Baseline MCM vs Attention-Based MCM\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input Dimension: {input_dim}\n")
        f.write(f"Number of Masks: {num_masks}\n")
        f.write(f"Training Epochs: {num_epochs}\n")
        f.write(f"Test Samples: {len(X_test_np)}\n")
        f.write(f"Anomaly Ratio: {y_test_np.mean():.2%}\n\n")

        f.write("RESULTS:\n")
        f.write("-" * 70 + "\n")
        f.write(df_results.to_string(index=False) + "\n\n")

        f.write("IMPROVEMENT:\n")
        f.write("-" * 70 + "\n")
        f.write(f"AUC-ROC: {improvement_roc:+.2f}%\n")
        f.write(f"AUC-PR:  {improvement_pr:+.2f}%\n\n")

        f.write("MODEL CONFIGURATIONS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Baseline MCM:\n")
        f.write(f"  - Mask Type: Static/Learnable\n")
        f.write(f"  - Parameters: {sum(p.numel() for p in model_baseline.parameters()):,}\n\n")

        f.write(f"Attention-Based MCM:\n")
        f.write(f"  - Mask Type: Self-Attention\n")
        f.write(f"  - Attention Heads: 4\n")
        f.write(f"  - Parameters: {sum(p.numel() for p in model_attention.parameters()):,}\n")

    print(f"Detailed report saved to: {report_path}")
    try:
        plot_comparison(all_results, dataset_name, output_dir)
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()