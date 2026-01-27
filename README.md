# FedMTFI

A comprehensive federated learning framework implementing cluster-based federated learning with multi-teacher knowledge distillation and SHAP-based feature importance optimization.

## Overview

FedMTFI is a federated learning system that combines:
- **Cluster-based Federated Learning**: Organizes clients into clusters with specialized model architectures
- **Multi-Teacher Knowledge Distillation (MTKD)**: Uses multiple cluster prototype models as teachers to train a unified global student model
- **SHAP-based Feature Importance Optimization**: Incorporates Shapley value-based feature importance weighting to enhance learning efficiency and model interpretability
- **Dirichlet-based Non-IID Data Distribution**: Handles realistic heterogeneous data scenarios with configurable heterogeneity levels

## Key Features

### Core Architecture
- **Cluster-Specific Models**: SimpleCNN, ResNet-like, MobileNet-like, and ResNet18-like architectures
- **Adaptive Input Handling**: Supports both grayscale (MNIST, Fashion-MNIST) and RGB (CIFAR-10) datasets
- **Multi-Teacher Distillation**: Aggregates knowledge from cluster models into a unified student model
- **Post-hoc Knowledge Distillation**: Final refinement phase using trained cluster models as teachers

### Federated Learning Approaches
1. **FedMTFI (Main)**: Novel cluster-based approach with multi-teacher knowledge distillation and SHAP weighting
2. **FedAvg**: Traditional federated averaging implementation using Flower framework
3. **FedProx**: Proximal federated learning with regularization terms
4. **Centralized**: Baseline centralized learning for comparison

### Advanced Features
- **SHAP Toggle**: Enable/disable SHAP-based feature importance for ablation studies
- **Dirichlet Alpha Configuration**: Control non-IID data heterogeneity (α = 0.1 highly non-IID, α = 10.0 near IID)
- **Interactive Configuration**: User-friendly setup for experiment parameters
- **Comprehensive Metrics**: Detailed logging of training, evaluation, and timing metrics
- **Visualization**: Automated plot generation for training progress and results

## Project Structure

```
FedMTFI-main/
├── main.py                    # Main FedMTFI implementation
├── config.py                  # Configuration management with SHAP toggle and Dirichlet alpha
├── models.py                  # Neural network architectures
├── client.py                  # Client-side training logic
├── server.py                  # Server-side aggregation and distillation
├── distillation.py            # Knowledge distillation utilities
├── non_iid_distributor.py     # Dirichlet-based data distribution strategies
├── metrics_logger.py          # Comprehensive metrics tracking
├── shap_utils.py              # Feature importance analysis
├── plotting_utils.py          # Visualization utilities
├── run_ablation_study.py      # Ablation study runner (SHAP vs no SHAP)
├── run_with_shap.py           # Run experiment with SHAP enabled
├── run_without_shap.py        # Run experiment without SHAP (ablation)
├── demo_shap_toggle.py        # SHAP toggle demonstration script
├── FedAvg/                    # FedAvg baseline implementations
│   ├── CIFAR-10/
│   ├── FMNIST/
│   └── MNIST/
├── FedProx/                   # Proximal federated learning
├── Centralized/               # Baseline centralized learning
└── data/                      # Dataset storage
```

## Supported Datasets

- **MNIST**: Handwritten digits (28x28 grayscale) - Private client training data
- **Fashion-MNIST**: Fashion items (28x28 grayscale) - Public evaluation dataset
- **CIFAR-10**: Natural images (32x32 RGB) - Public evaluation dataset

All datasets are automatically downloaded and preprocessed with appropriate transformations.

## Model Architectures

### Cluster Models (Teachers)
1. **Cluster 0 - SimpleCNN**: Lightweight CNN with ~0.8M parameters for resource-constrained devices
2. **Cluster 1 - ResNet-like**: Residual connections with ~1.5M parameters, balancing performance and efficiency
3. **Cluster 2 - MobileNet-like**: Depthwise separable convolutions with ~1.2M parameters for mobile/edge devices
4. **Cluster 3 - ResNet18-like**: Full ResNet-18 inspired architecture with ~2.1M parameters for capable devices

### Student Model
- Compact StudentCNN architecture (~0.3M parameters) for knowledge distillation
- Adaptive input channels based on dataset
- Optimized for multi-teacher learning

## Installation

### Prerequisites
```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install tensorflow  # For centralized baselines
pip install shap captum  # For feature importance
pip install openpyxl  # For Excel metrics export
```

### Quick Start
```bash
git clone https://github.com/shadhin39/FedMTFI.git
cd FedMTFI-main
python main.py
```

## Usage

### Main FedMTFI Experiment
```bash
python main.py
```

The system will prompt for configuration parameters:
- Number of clients (default: 100)
- Number of clusters (default: 4)
- Clients per round (default: 20)
- Training rounds (default: 30)
- Local epochs (default: 5)
- **Dirichlet alpha for non-IID distribution (default: 0.5)**
- Distillation epochs for each dataset

### FedAvg Experiments
```bash
# CIFAR-10
cd FedAvg/CIFAR-10
python server.py  # Terminal 1
python client1.py  # Terminal 2
python client2.py  # Terminal 3

# Fashion-MNIST
cd FedAvg/FMNIST
python server.py  # Terminal 1
python client1.py  # Terminal 2
python client2.py  # Terminal 3
```

### FedProx Experiments
```bash
cd FedProx

# Fashion-MNIST experiment
python3 image_main.py --dataset FMNIST --r 5 --E 3 --K 10 --C 0.3 --B 32 --lr 0.01 --mu 0.01 --alpha 0.5 --model_type simple --device cpu

# CIFAR-10 experiment
python3 image_main.py --dataset CIFAR10 --r 5 --E 3 --K 10 --C 0.3 --B 32 --lr 0.01 --mu 0.01 --alpha 0.5 --model_type simple --device cpu
```

### Centralized Baselines
```bash
cd Centralized/CIFAR-10
python centralized_cifar10.py

cd Centralized/FMNIST
python centralized_fminst.py
```

## Configuration

### SHAP Toggle (Ablation Study)

Enable or disable SHAP-based feature importance weighting in `config.py`:
```python
CFG.use_shap = True   # Enable SHAP-based feature importance (default)
CFG.use_shap = False  # Disable SHAP weighting for ablation study
```

When SHAP is enabled:
- Loss function: `L_weighted = φ̄ · L_total` (Shapley-weighted)
- Feature importance is computed for each teacher model
- Distillation prioritizes informative features

When SHAP is disabled:
- Loss function: `L_total` (standard KD + CE + feature loss)
- No Shapley value computation
- Useful for baseline comparison

### Dirichlet Alpha (Non-IID Configuration)

Control data heterogeneity using the Dirichlet distribution parameter in `config.py`:
```python
CFG.alpha_dirichlet = 0.5  # Standard non-IID (default)
```

| Alpha Value | Heterogeneity Level | Description |
|-------------|---------------------|-------------|
| 0.1 | Highly non-IID | Clients have very skewed label distributions |
| 0.3 | Moderately non-IID | Noticeable label imbalance |
| 0.5 | Standard non-IID | Realistic federated scenario |
| 1.0 | Mild non-IID | Most labels present but uneven |
| 10.0 | Near IID | Close to uniform distribution |

### Interactive Configuration

The main system provides interactive configuration:
```bash
python main.py
# Prompts for: clients, clusters, rounds, epochs, Dirichlet alpha, etc.
```

### Programmatic Configuration

Modify `config.py` for batch experiments:
```python
CFG.set_federated_config(
    num_clients=50,
    num_clusters=3,
    rounds=20,
    local_epochs=3,
    alpha_dirichlet=0.5  # Non-IID distribution parameter
)
```

### Dataset-Specific Parameters
```python
CFG.dataset_configs = {
    "FashionMNIST": {
        "batch_size": 64,
        "lr_server": 0.001,
        "distill_epochs": 5,
        "temperature": 3.0
    },
    "CIFAR10": {
        "batch_size": 128,
        "lr_server": 0.0005,
        "distill_epochs": 10,
        "temperature": 5.0
    }
}
```

## Key Algorithms

### Cluster-Based Federated Learning
1. **Client Assignment**: Round-robin assignment to clusters
2. **Local Training**: Clients train cluster-specific models on private non-IID data (partitioned via Dirichlet)
3. **Signal Generation**: Clients produce knowledge signals on public data
4. **Cluster Aggregation**: Server aggregates client models using FedAvg to create cluster prototypes
5. **Model Update**: Updated cluster models distributed to clients

### Multi-Teacher Knowledge Distillation
1. **Teacher Training**: Cluster prototype models trained on public datasets
2. **Confidence Weighting**: Teachers weighted by prediction confidence
3. **SHAP-based Feature Importance**: Shapley values computed to weight distillation loss
4. **Combined Loss**: `L_total = (1-α)·L_CE + α·T²·L_KD`, optionally weighted by SHAP

### Non-IID Data Distribution (Dirichlet)
The framework uses Dirichlet distribution to create realistic non-IID data partitions:
```python
# Partition data using Dirichlet distribution
partitions = dirichlet_partition(dataset, num_clients, alpha=0.5)
```
Lower α values create more heterogeneous (skewed) distributions.

## Ablation Study: SHAP Weighting

### Running the Ablation Study
```bash
# Run with SHAP enabled
python run_ablation_study.py --with-shap

# Run without SHAP (ablation)
python run_ablation_study.py --without-shap

# Run both experiments for comparison
python run_ablation_study.py --both

# With custom configuration
python run_ablation_study.py --both --clusters 3 --clients 10 --rounds 30
```

### Alternative Scripts
```bash
# Dedicated scripts for each configuration
python run_with_shap.py     # Full experiment with SHAP
python run_without_shap.py  # Full experiment without SHAP

# Quick demonstration of SHAP toggle
python demo_shap_toggle.py
```

### Expected Results
Based on experiments with 3 clusters, 10 clients, 30 rounds, α=0.5:

| Configuration | CIFAR-10 (%) | FMNIST (%) |
|--------------|-------------|------------|
| FedMTFI w/o SHAP | 61.82 | 85.43 |
| FedMTFI w/ SHAP | **64.48** | **87.28** |
| **Improvement** | +2.66 | +1.85 |

### Dirichlet Alpha Sensitivity (MNIST Client Accuracy)

| Alpha | Accuracy (%) | Heterogeneity |
|-------|-------------|---------------|
| 0.1 | 91.24 | Highly non-IID |
| 0.3 | 93.18 | Moderately non-IID |
| 0.5 | 94.56 | Standard non-IID |
| 1.0 | 96.12 | Mild non-IID |
| 10.0 | 97.85 | Near IID |

## Experimental Comparisons

The framework enables comprehensive comparison between:
- **FedMTFI vs FedAvg**: Cluster-based MTKD vs traditional averaging
- **FedMTFI vs FedProx**: Knowledge distillation vs proximal terms
- **FedMTFI vs Centralized**: Federated vs centralized learning
- **SHAP vs No-SHAP**: Ablation study on feature importance weighting
- **IID vs Non-IID**: Impact of Dirichlet alpha on performance

## Troubleshooting

### Common Issues
1. **CUDA Memory**: Reduce batch sizes in `config.py`
2. **Model Compatibility**: Ensure consistent input channels across datasets
3. **Data Loading**: Verify dataset paths and download permissions
4. **Flower Connection**: Check network settings for FedAvg experiments
5. **SHAP Computation**: If SHAP is slow, use gradient-based approximation (default)

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Reduce problem size for testing
CFG.num_clients = 10
CFG.rounds = 5
CFG.local_epochs = 2
CFG.use_shap = False  # Disable SHAP for faster testing
```


## License

This project is licensed under the MIT License.
