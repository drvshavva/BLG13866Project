import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from model.trainer import ModelTrainer

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
        'data_dim': 9,
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
    'small': ['thyroid', 'mammography', 'pima', 'breastw', 'shuttle', 'glass', 'wine', 'pendigits'],
    'medium': ['cardio', 'cardiotocography', 'fraud', 'wbc', 'ionosphere', 'satellite', 'satimage-2', 'campaign', 'optdigits'],
    'large': ['census'],
}

DATASET_CATEGORIES['all'] = (
    DATASET_CATEGORIES['small'] +
    DATASET_CATEGORIES['medium'] +
    DATASET_CATEGORIES['large']
)

DEFAULT_SETTINGS = {
    'epochs': 200,
    'learning_rate': 0.05,
    'sche_gamma': 0.98,
    'lambda': 5.0,
    'device': 'cuda',
    'batch_size': 512,
    'en_nlayers': 3,
    'de_nlayers': 3,
    'mask_nlayers': 3,
    'data_dir': 'datasets/',
    'num_workers': 0,
    'random_seed': 42,
    'runs': 1,
    'validate_every': 10,
    'early_stopping': False,
    'patience': 20,
    'weight_decay': 1e-5,
    'optimizer': 'adam',
    'scheduler_type': 'exponential',
    'dropout': 0.0,
    'grad_clip': 0,
}


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_config(dataset_name: str, custom_settings: Dict = None) -> Dict:
    config = DEFAULT_SETTINGS.copy()

    if dataset_name in DATASET_CONFIGS:
        config.update(DATASET_CONFIGS[dataset_name])
    else:
        raise ValueError(f"Bilinmeyen dataset: {dataset_name}")

    config['dataset_name'] = dataset_name

    if custom_settings:
        config.update(custom_settings)

    return config


def analyze_model_explainability(
    trainer: ModelTrainer,
    results_dir: Path,
    num_samples: int = 100
):
    """
    Model açıklanabilirliği analizi yap

    Analiz edilen:
    - Mask önem dereceleri
    - Feature önem dereceleri
    - Anomali örnekleri
    - Mask çeşitliliği
    """
    print("\n" + "="*70)
    print("Model Açıklanabilirliği Analizi")
    print("="*70)

    trainer.model.eval()

    all_features = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for i, (x_input, y_label) in enumerate(trainer.test_loader):
            if i * trainer.test_loader.batch_size >= num_samples:
                break

            x_input = x_input.to(trainer.device)
            x_pred, latent, masks = trainer.model(x_input)
            scores = trainer.scorer(x_input, x_pred)

            all_features.append(x_input.cpu())
            all_labels.append(y_label)
            all_scores.append(scores.cpu())

    features = torch.cat(all_features, dim=0)[:num_samples]
    labels = torch.cat(all_labels, dim=0)[:num_samples]
    scores = torch.cat(all_scores, dim=0)[:num_samples]

    # 1. Mask Analizi
    print("\n1. Mask Çeşitliliği ve Önem Analizi")
    analyze_masks(trainer, features, results_dir)

    # 2. Feature Önem Analizi
    print("\n2. Feature Önem Analizi")
    analyze_feature_importance(trainer, features, labels, results_dir)

    # 3. Anomali Örnekleri
    print("\n3. En Yüksek Skorlu Anomali Örnekleri")
    explain_top_anomalies(trainer, features, labels, scores, results_dir, top_k=5)

    # 4. Normal vs Anomali Karşılaştırması
    print("\n4. Normal vs Anomali Karşılaştırması")
    compare_normal_vs_anomaly(trainer, features, labels, scores, results_dir)

    print("\nAçıklanabilirlik analizi tamamlandı!")


def analyze_masks(trainer: ModelTrainer, features: torch.Tensor, results_dir: Path):
    """Mask çeşitliliği ve önem derecelerini analiz et"""
    with torch.no_grad():
        features = features.to(trainer.device)

        # Mask istatistikleri
        mask_stats = trainer.model.get_mask_statistics(features)

        print(f"  Ortalama mask çeşitliliği: {mask_stats['diversity']:.4f}")
        print(f"\n  Mask bazında istatistikler:")

        for i in range(trainer.config['mask_num']):
            print(f"    Mask {i+1}:")
            print(f"      Aktivasyon ortalaması: {mask_stats['mean'][i]:.3f}")
            print(f"      Seyreltme (sparsity):  {mask_stats['sparsity'][i]:.2%}")

        # Görselleştirme
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Mask aktivasyon ortalamaları
        axes[0].bar(range(1, len(mask_stats['mean'])+1), mask_stats['mean'].cpu().numpy())
        axes[0].set_xlabel('Mask ID')
        axes[0].set_ylabel('Ortalama Aktivasyon')
        axes[0].set_title('Mask Aktivasyon Seviyeleri')
        axes[0].grid(True, alpha=0.3)

        # Mask seyreltme oranları
        axes[1].bar(range(1, len(mask_stats['sparsity'])+1), mask_stats['sparsity'].cpu().numpy())
        axes[1].set_xlabel('Mask ID')
        axes[1].set_ylabel('Seyreltme Oranı')
        axes[1].set_title('Mask Seyreltme (Sparsity)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(results_dir / 'mask_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n  [OK] Görselleştirme kaydedildi: mask_analysis.png")


def analyze_feature_importance(
    trainer: ModelTrainer,
    features: torch.Tensor,
    labels: torch.Tensor,
    results_dir: Path
):
    """Feature önem derecelerini analiz et"""
    with torch.no_grad():
        features = features.to(trainer.device)

        # Reconstruction error'ları feature bazında hesapla
        x_pred, _, _ = trainer.model(features)
        feature_errors = trainer.scorer.compute_feature_wise_errors(features, x_pred)

        # Normal ve anomali örnekleri için ayrı hesapla
        normal_mask = labels == 0
        anomaly_mask = labels == 1

        if normal_mask.sum() > 0 and anomaly_mask.sum() > 0:
            normal_errors = feature_errors[normal_mask].mean(dim=(0, 1)).cpu().numpy()
            anomaly_errors = feature_errors[anomaly_mask].mean(dim=(0, 1)).cpu().numpy()

            # Fark = Anomalilerde daha yüksek olan feature'lar
            importance = anomaly_errors - normal_errors

            # En önemli 10 feature
            top_indices = np.argsort(importance)[-10:][::-1]

            print("  En önemli 10 feature (anomali tespiti için):")
            for rank, idx in enumerate(top_indices, 1):
                print(f"    {rank}. Feature {idx}: önem={importance[idx]:.4f}")

            # Görselleştirme
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Feature önem dereceleri
            axes[0].barh(range(len(top_indices)), importance[top_indices])
            axes[0].set_yticks(range(len(top_indices)))
            axes[0].set_yticklabels([f'Feature {i}' for i in top_indices])
            axes[0].set_xlabel('Önem Derecesi (Anomali - Normal)')
            axes[0].set_title('En Önemli Features')
            axes[0].grid(True, alpha=0.3, axis='x')

            # Tüm feature'lar için heatmap
            axes[1].plot(range(len(importance)), importance, 'o-', markersize=3, alpha=0.6)
            axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1].set_xlabel('Feature Index')
            axes[1].set_ylabel('Önem Derecesi')
            axes[1].set_title('Tüm Features için Önem Dağılımı')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(results_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()

            # JSON olarak kaydet
            importance_dict = {
                'all_features': importance.tolist(),
                'top_10_features': [
                    {'feature_id': int(idx), 'importance': float(importance[idx])}
                    for idx in top_indices
                ]
            }

            with open(results_dir / 'feature_importance.json', 'w') as f:
                json.dump(importance_dict, f, indent=2)

            print(f"\n  [OK] Feature importance kaydedildi")


def explain_top_anomalies(
    trainer: ModelTrainer,
    features: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    results_dir: Path,
    top_k: int = 5
):
    """En yüksek skorlu anomalileri açıkla"""
    scores_np = scores.squeeze().numpy()

    # En yüksek skorlu örnekleri bul
    top_indices = np.argsort(scores_np)[-top_k:][::-1]

    explanations = []

    for rank, idx in enumerate(top_indices, 1):
        sample = features[idx:idx+1].to(trainer.device)
        label = labels[idx].item()
        score = scores_np[idx]

        top_features = trainer.scorer.get_top_anomalous_features(
            sample,
            trainer.model(sample)[0],
            top_k=5
        )

        explanation = {
            'rank': rank,
            'sample_id': int(idx),
            'anomaly_score': float(score),
            'true_label': 'Anomali' if label == 1 else 'Normal',
            'top_anomalous_features': top_features.cpu().numpy().tolist()[0]
        }

        explanations.append(explanation)

        print(f"  #{rank} - Skor: {score:.4f} | Gerçek: {explanation['true_label']}")
        print(f"       En anomali features: {explanation['top_anomalous_features'][:3]}")

    with open(results_dir / 'top_anomalies_explained.json', 'w') as f:
        json.dump(explanations, f, indent=2)

    print(f"\n  [OK] Anomali açıklamaları kaydedildi")


def compare_normal_vs_anomaly(
    trainer: ModelTrainer,
    features: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    results_dir: Path
):
    """Normal ve anomali örneklerini karşılaştır"""
    normal_mask = labels == 0
    anomaly_mask = labels == 1

    if normal_mask.sum() == 0 or anomaly_mask.sum() == 0:
        print("  [UYARI] Normal veya anomali örneği yok, karşılaştırma yapılamıyor")
        return

    normal_scores = scores[normal_mask].squeeze().numpy()
    anomaly_scores = scores[anomaly_mask].squeeze().numpy()

    print(f"  Normal örnekler:  n={len(normal_scores)}, skor ort={normal_scores.mean():.4f}")
    print(f"  Anomali örnekler: n={len(anomaly_scores)}, skor ort={anomaly_scores.mean():.4f}")

    # Görselleştirme
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(normal_scores, bins=30, alpha=0.6, label='Normal', color='blue')
    axes[0].hist(anomaly_scores, bins=30, alpha=0.6, label='Anomali', color='red')
    axes[0].set_xlabel('Anomali Skoru')
    axes[0].set_ylabel('Frekans')
    axes[0].set_title('Skor Dağılımı Karşılaştırması')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    axes[1].boxplot([normal_scores, anomaly_scores], labels=['Normal', 'Anomali'])
    axes[1].set_ylabel('Anomali Skoru')
    axes[1].set_title('Skor İstatistikleri')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(results_dir / 'normal_vs_anomaly.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  [OK] Karşılaştırma görselleştirmesi kaydedildi")


def run_single_dataset(
    dataset_name: str,
    custom_settings: Dict = None,
    enable_explainability: bool = True
) -> Dict:
    """
    Tek bir dataset üzerinde model eğit ve değerlendir

    Args:
        dataset_name: Veri kümesi adı
        custom_settings: Özel ayarlar (opsiyonel)
        enable_explainability: Açıklanabilirlik analizi yapılsın mı

    Returns:
        Sonuç metrikleri
    """
    print("\n" + "="*70)
    print(f"Dataset: {dataset_name.upper()}")
    print("="*70)

    # Konfigürasyon oluştur
    config = create_config(dataset_name, custom_settings)

    # Seed ayarla
    setup_seed(config['random_seed'])

    # Sonuç klasörü oluştur
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('./results') / dataset_name / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Konfigürasyonu kaydet
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Trainer oluştur
    print("\n[1/3] Model hazırlanıyor...")
    trainer = ModelTrainer(
        model_config=config,
        run_id=0,
        checkpoint_dir=results_dir / 'checkpoints',
        log_dir=results_dir / 'logs'
    )

    # Eğitim
    print("\n[2/3] Eğitim başlıyor...")
    trainer.train(
        num_epochs=config['epochs'],
        validate_every=config['validate_every'],
        early_stopping=config['early_stopping'],
        patience=config.get('patience', 20),
        save_best=True
    )

    # Değerlendirme
    print("\n[3/3] Model değerlendiriliyor...")
    checkpoint_path = trainer.checkpoint_dir / 'best_model.pth'
    if not checkpoint_path.exists():
        checkpoint_path = None

    metrics = trainer.evaluate(checkpoint_path=checkpoint_path)

    # Açıklanabilirlik analizi
    if enable_explainability:
        analyze_model_explainability(trainer, results_dir)

    # Sonuçları kaydet
    results = {
        'dataset': dataset_name,
        'metrics': metrics,
        'config': config,
        'timestamp': timestamp
    }

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Sonuçlar kaydedildi: {results_dir}")
    print(f"\n  AUC-ROC: {metrics['roc_auc']:.4f}")
    print(f"  AUC-PR:  {metrics['average_precision']:.4f}")
    print(f"  F1:      {metrics['f1_score']:.4f}")

    return results


def run_multiple_datasets(
    dataset_names: List[str],
    custom_settings: Dict = None,
    enable_explainability: bool = True
) -> Dict:
    """
    Birden fazla dataset üzerinde çalıştır ve karşılaştır

    Args:
        dataset_names: Dataset isimleri listesi
        custom_settings: Tüm datasetler için özel ayarlar
        enable_explainability: Açıklanabilirlik analizi yapılsın mı

    Returns:
        Tüm sonuçlar
    """
    print("\n" + "="*70)
    print("ÇOKLU DATASET ANALİZİ")
    print("="*70)
    print(f"Datasets: {', '.join(dataset_names)}")

    all_results = {}

    for dataset_name in dataset_names:
        try:
            results = run_single_dataset(
                dataset_name,
                custom_settings,
                enable_explainability
            )
            all_results[dataset_name] = results
        except Exception as e:
            print(f"\n[HATA] {dataset_name} için hata oluştu: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Karşılaştırmalı sonuçlar
    if len(all_results) > 1:
        print_comparison_results(all_results)
        save_comparison_results(all_results)

    return all_results


def print_comparison_results(all_results: Dict):
    """Karşılaştırmalı sonuçları yazdır"""
    print("\n" + "="*70)
    print("KARŞILAŞTIRMALI SONUÇLAR")
    print("="*70)

    print(f"\n{'Dataset':<15} {'AUC-ROC':<10} {'AUC-PR':<10} {'F1 Score':<10}")
    print("-" * 70)

    for dataset, results in all_results.items():
        metrics = results['metrics']
        print(f"{dataset:<15} "
              f"{metrics['roc_auc']:<10.4f} "
              f"{metrics['average_precision']:<10.4f} "
              f"{metrics['f1_score']:<10.4f}")


def save_comparison_results(all_results: Dict):
    """Karşılaştırmalı sonuçları kaydet"""
    comparison_dir = Path('./results/comparison')
    comparison_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON olarak kaydet
    with open(comparison_dir / f'comparison_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Görselleştirme
    datasets = list(all_results.keys())
    auc_roc = [all_results[d]['metrics']['roc_auc'] for d in datasets]
    auc_pr = [all_results[d]['metrics']['average_precision'] for d in datasets]
    f1 = [all_results[d]['metrics']['f1_score'] for d in datasets]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(datasets, auc_roc, color='skyblue')
    axes[0].set_ylabel('AUC-ROC')
    axes[0].set_title('ROC-AUC Karşılaştırması')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(datasets, auc_pr, color='lightcoral')
    axes[1].set_ylabel('AUC-PR')
    axes[1].set_title('Average Precision Karşılaştırması')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(datasets, f1, color='lightgreen')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score Karşılaştırması')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(comparison_dir / f'comparison_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    try:
        run_multiple_datasets(DATASET_CATEGORIES['all'], enable_explainability=False)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)