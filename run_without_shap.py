"""
Run FedMTFI WITHOUT SHAP (ablation baseline, non-interactive version).
This script runs the full training pipeline WITHOUT SHAP-based feature importance weighting.
"""

import sys
import torch
import random
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import CFG
from client import Client
from server import ClusterServer
from models import build_adaptive_model
from non_iid_distributor import NonIIDDataDistributor
from metrics_logger import MetricsLogger


def assign_clients_to_clusters(num_clients: int, num_clusters: int):
    """Assign clients to clusters in a round-robin fashion."""
    assignments = {}
    for client_id in range(num_clients):
        cluster_id = client_id % num_clusters
        assignments[client_id] = cluster_id
    return assignments


def select_random_clients(clients, clients_per_round: int):
    """Randomly select a subset of clients for the current round."""
    if clients_per_round >= len(clients):
        return clients
    return random.sample(clients, clients_per_round)


def load_datasets():
    """Load and prepare datasets for training and evaluation."""
    grayscale_to_rgb_transform = transforms.Compose([
        transforms.Resize((CFG.image_size, CFG.image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    rgb_transform = transforms.Compose([
        transforms.Resize((CFG.image_size, CFG.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    datasets_dict = {}
    
    fashion_train = datasets.FashionMNIST(
        root=CFG.data_dir, train=True, download=True, transform=grayscale_to_rgb_transform
    )
    fashion_test = datasets.FashionMNIST(
        root=CFG.data_dir, train=False, download=True, transform=grayscale_to_rgb_transform
    )
    datasets_dict["FashionMNIST"] = {"train": fashion_train, "test": fashion_test}
    
    mnist_train = datasets.MNIST(
        root=CFG.data_dir, train=True, download=True, transform=grayscale_to_rgb_transform
    )
    mnist_test = datasets.MNIST(
        root=CFG.data_dir, train=False, download=True, transform=grayscale_to_rgb_transform
    )
    datasets_dict["MNIST"] = {"train": mnist_train, "test": mnist_test}
    
    cifar10_train = datasets.CIFAR10(
        root=CFG.data_dir, train=True, download=True, transform=rgb_transform
    )
    cifar10_test = datasets.CIFAR10(
        root=CFG.data_dir, train=False, download=True, transform=rgb_transform
    )
    datasets_dict["CIFAR10"] = {"train": cifar10_train, "test": cifar10_test}
    
    return datasets_dict


def main_without_shap():
    """Main training function with SHAP DISABLED (ablation baseline)."""
    
    # Configuration
    num_clients = 10
    num_clusters = 3
    clients_per_round = 10
    rounds = 5
    local_epochs = 5
    
    # DISABLE SHAP for ablation study
    CFG.use_shap = False
    
    # Set configuration programmatically
    CFG.set_federated_config(
        num_clients=num_clients,
        num_clusters=num_clusters,
        clients_per_round=clients_per_round,
        rounds=rounds,
        local_epochs=local_epochs,
        fmnist_distill_epochs=5,
        cifar10_distill_epochs=10,
        fmnist_student_epochs=15,
        cifar10_student_epochs=20
    )
    
    print("=" * 60)
    print("FedMTFI - WITHOUT SHAP (ABLATION BASELINE)")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - SHAP weighting: {'ENABLED' if CFG.use_shap else 'DISABLED'}")
    print(f"  - Clients: {CFG.num_clients}")
    print(f"  - Clusters: {CFG.num_clusters}")
    print(f"  - Rounds: {CFG.rounds}")
    print(f"  - Local epochs: {CFG.local_epochs}")
    print("=" * 60)
    
    overall_start_time = time.time()
    
    device = torch.device(CFG.device if torch.cuda.is_available() else "cpu")
    print(f"[FedMTFI] Using device: {device}")
    
    # Load datasets
    print("[FedMTFI] Loading datasets...")
    datasets_dict = load_datasets()
    
    private_dataset = datasets_dict["MNIST"]
    public_datasets = {
        "FashionMNIST": datasets_dict["FashionMNIST"],
        "CIFAR10": datasets_dict["CIFAR10"]
    }
    
    # Create public loaders
    public_loaders = {}
    for dataset_name, dataset in public_datasets.items():
        config = CFG.dataset_configs[dataset_name]
        public_loaders[dataset_name] = DataLoader(
            dataset["train"], 
            batch_size=config["batch_size"], 
            shuffle=True
        )
        print(f"[FedMTFI] Created public dataset ({dataset_name}) with {len(dataset['train'])} samples")
    
    public_loader = public_loaders["FashionMNIST"]
    
    # Distribute private data (non-IID)
    print("[FedMTFI] Distributing private data (MNIST) to clients (non-IID)...")
    distributor = NonIIDDataDistributor(
        dataset=private_dataset["train"],
        num_clients=CFG.num_clients,
        num_classes=10
    )
    client_datasets, client_preferences = distributor.bias_based_distribution(primary_bias=0.8)
    
    # Assign clients to clusters
    cluster_assignments = assign_clients_to_clusters(CFG.num_clients, CFG.num_clusters)
    print(f"[FedMTFI] Client-cluster assignments: {cluster_assignments}")
    
    # Initialize server
    server = ClusterServer(
        device=device,
        num_clusters=CFG.num_clusters,
        num_classes=CFG.num_classes,
        in_channels=CFG.in_channels,
        image_size=CFG.image_size,
        dataset_name="CIFAR10"
    )
    
    # Build clients
    print("[FedMTFI] Building clients...")
    clients = []
    for client_id in range(CFG.num_clients):
        cluster_id = cluster_assignments[client_id]
        model = build_adaptive_model(cluster_id, "CIFAR10", CFG.num_classes, CFG.image_size)
        client_loader = DataLoader(
            client_datasets[client_id], 
            batch_size=CFG.batch_size, 
            shuffle=True
        )
        
        client = Client(
            cid=client_id,
            model=model,
            train_loader=client_loader,
            device=device,
            cluster_id=cluster_id
        )
        clients.append(client)
        print(f"[FedMTFI] Client {client_id} -> Cluster {cluster_id}")
    
    # Initialize logger
    logger = MetricsLogger()
    
    print(f"[FedMTFI] Starting federated learning for {CFG.rounds} rounds...")
    
    # Federated learning rounds
    for round_num in range(1, CFG.rounds + 1):
        print(f"\n[FedMTFI] ===== Round {round_num}/{CFG.rounds} =====")
        round_start_time = time.time()
        
        selected_clients = select_random_clients(clients, CFG.clients_per_round)
        print(f"[FedMTFI] Selected {len(selected_clients)} clients")
        
        client_signals = []
        
        for client in selected_clients:
            print(f"[FedMTFI] Training client {client.id}...")
            local_stats, training_time = client.train_local(epochs=CFG.local_epochs, round_num=round_num, logger=logger)
            signal = client.produce_signals(public_loader, current_round=round_num)
            signal["cluster_id"] = client.cluster_id
            client_signals.append(signal)
            logger.add_local(local_stats)
        
        # Server-side training
        print(f"[FedMTFI] Training cluster models...")
        cluster_stats = server.train_cluster_models(
            client_signals, 
            public_loader, 
            epochs=CFG.cluster_distill_epochs,
            current_round=round_num,
            logger=logger
        )
        
        logger.add_distill(cluster_stats)
        
        # Update clients
        for client in selected_clients:
            cluster_model = server.cluster_models[client.cluster_id]
            client.model.load_state_dict(cluster_model.state_dict())
        
        round_duration = time.time() - round_start_time
        logger.log_round_time(round_num, round_duration)
        print(f"[FedMTFI] Round {round_num} completed in {round_duration:.2f}s")
    
    # Post-hoc distillation
    print(f"\n[FedMTFI] ===== Post-hoc Knowledge Distillation =====")
    
    for dataset_name in ["FashionMNIST", "CIFAR10"]:
        print(f"\n[FedMTFI] Training on {dataset_name}...")
        
        train_loader = public_loaders[dataset_name]
        test_loader = DataLoader(
            public_datasets[dataset_name]["test"],
            batch_size=CFG.eval_batch_size,
            shuffle=False
        )
        
        # Train cluster models
        server.train_cluster_models_on_public_dataset(
            train_loader,
            dataset_name=dataset_name,
            epochs=3,  # Reduced for demo
            logger=logger
        )
        
        # Multi-teacher distillation (NO SHAP WEIGHTING)
        student_stats = server.train_student_with_teachers(
            train_loader,
            dataset_name=dataset_name,
            epochs=3,  # Reduced for demo
            current_round=-1,
            logger=logger,
            client_signals=None
        )
        
        # Evaluate
        student_acc = server.evaluate_student(test_loader)
        print(f"[FedMTFI] Student accuracy on {dataset_name}: {student_acc*100:.2f}%")
        
        eval_record = {
            "round": -1,
            "dataset": dataset_name.lower(),
            "model": "student_posthoc",
            "accuracy": student_acc
        }
        logger.log_evaluation_metrics(eval_record)
    
    # Save results
    results_file = "federated_learning_results_WITHOUT_SHAP.xlsx"
    logger.save_excel(results_file)
    print(f"\n[FedMTFI] Results saved to {results_file}")
    
    overall_duration = time.time() - overall_start_time
    logger.log_overall_training_time(overall_duration)
    print(f"[FedMTFI] Total training time: {overall_duration:.2f}s ({overall_duration/60:.2f} min)")
    print(f"\n[FedMTFI] Training WITHOUT SHAP completed!")
    
    return student_acc


if __name__ == "__main__":
    main_without_shap()
