# MCM: Masked Cell Modeling for Anomaly Detection In Tabular Data

This repository contains the Python implementation of the MCM paper: [MCM: Masked Cell Modeling for Anomaly Detection in Tabular Data](https://openreview.net/forum?id=lNZJyEDxy4) published at ICLR 2024 as a conference paper.

**Paper:** "MCM: Masked Cell Modeling for Anomaly Detection in Tabular Data"  
**Authors:** Jiaxin Yin, Yuanyuan Qiao, Zitang Zhou, Xiangchao Wang, Jie Yang  
**Venue:** ICLR 2024

## What is MCM?
MCM (Masked Cell Modeling) is a self-supervised learning method designed for anomaly detection on tabular data.

## Core idea
- Bring the success of MAE (Masked Autoencoder) from computer vision and
- BERT (Masked Language Modeling) from NLP into tabular anomaly detection

## How it works
1. Learn the correlations among normal data
   - If Feature A is high → Feature B should also be high
   - These correlations represent "normal behavior" patterns
2. At test time detect deviations from these correlations
   - Anomalies do not follow these correlations
   - Reconstruction error → anomaly score

## Key innovations
1. Learnable Masking — learn masks with a neural network instead of random masking
2. Multiple Masks — use K different masks (default K = 15)
3. Diversity Loss — encourage masks to be different from each other

## Mathematical formulation

**Toplam Loss:** $\mathcal{L} = \mathcal{L}_{rec} + \lambda \cdot \mathcal{L}_{div}$

**Reconstruction Loss:** $\mathcal{L}_{rec} = \frac{1}{K} \sum_{k=1}^{K} ||\hat{X}_k - X||^2$

**Diversity Loss:** $\mathcal{L}_{div} = \sum_{i=1}^{K} \left[ \ln \left( \sum_{j=1}^{K} e^{<M_i, M_j>/\tau} \cdot \mathbb{1}_{i \neq j} \right) \cdot scale \right]$

**Anomaly Score:** $score(x) = \frac{1}{K} \sum_{k=1}^{K} ||D(E(x \odot M_k)) - x||^2$