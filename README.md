# MCM: Masked Cell Modeling for Anomaly Detection In Tabular Data

This repository contains the Python implementation of the MCM paper: [MCM: Masked Cell Modeling for Anomaly Detection in Tabular Data](https://openreview.net/forum?id=lNZJyEDxy4) published at ICLR 2024 as a conference paper.

**Core Idea**

* Adapting the Masked Image/Language Modeling (MIM/MLM) approach to tabular data
* Detecting anomalies by learning the correlations among normal samples
* Using multiple masks to capture diverse types of correlations

**Architecture**

* Mask Generator: Produces learnable masks from the input data
* Encoder: Encodes the masked data into a latent space
* Decoder: Reconstructs the original data from the latent representation

**Loss Functions**

* Reconstruction Loss: Minimizes the reconstruction error
* Diversity Loss: Encourages diversity among different masks


## Prepare dataset
   1) When using your own data, move the dataset into `./Data`. 
   2) Add the dataset name to `./Dataset/DataLoader.py` based on the format of your dataset.
   3) Modify *dataset_name* and *data_dim* in `./main.py`
   4) You can download tabular datasets from [ODDS](https://odds.cs.stonybrook.edu/) and [ADBench](https://github.com/Minqi824/ADBench) for testing.

## Run
Run `main.py` to start training and testing the model. Results will be automatically stored in `./results`.


## Requirements
```
- Python 3.6
- PyTorch 1.10.2
- torchvision 0.11.3
- numpy 1.23.5
- pandas 1.5.3
- scipy 1.10.1
```
