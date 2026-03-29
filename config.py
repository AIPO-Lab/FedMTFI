"""
Configuration module for FedMTFI - Federated Multi-Task Feature Integration.

Provides the CFG class with all hyperparameters, dataset-specific configs,
and interactive/programmatic configuration methods.
"""

import torch


class CFG:
    """Global configuration for FedMTFI experiments."""

    # ── Device & paths ──────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "./data"
    seed = 42

    # ── Federated learning ──────────────────────────────────────────
    num_clients = 100
    num_clusters = 4
    clients_per_round = 20
    rounds = 30
    local_epochs = 5

    # ── Data ────────────────────────────────────────────────────────
    batch_size = 64
    eval_batch_size = 128
    num_classes = 10
    in_channels = 3          # unified 3-channel (grayscale converted to RGB)
    image_size = 32
    public_fraction = 0.5
    num_workers = 2

    # ── Optimiser ───────────────────────────────────────────────────
    lr_client = 0.001
    lr_server = 0.001
    weight_decay = 1e-4

    # ── Knowledge distillation ──────────────────────────────────────
    temperature = 3.0
    distill_epochs = 5
    cluster_distill_epochs = 5
    lambda_kd = 0.7
    lambda_feat = 0.1
    lambda_ce = 0.3

    # ── SHAP toggle ─────────────────────────────────────────────────
    use_shap = True

    # ── Non-IID distribution ────────────────────────────────────────
    alpha_dirichlet = 0.5

    # ── Evaluation datasets ─────────────────────────────────────────
    eval_datasets = ["FashionMNIST", "CIFAR10", "MNIST"]

    # ── Per-dataset configs ─────────────────────────────────────────
    dataset_configs = {
        "FashionMNIST": {
            "in_channels": 3,
            "image_size": 32,
            "batch_size": 64,
            "lr_server": 0.001,
            "distill_epochs": 5,
            "student_epochs": 15,
            "temperature": 3.0,
            "lambda_kd": 0.7,
            "lambda_feat": 0.1,
            "lambda_ce": 0.3,
            "normalization": {
                "mean": (0.5, 0.5, 0.5),
                "std": (0.5, 0.5, 0.5),
            },
        },
        "CIFAR10": {
            "in_channels": 3,
            "image_size": 32,
            "batch_size": 128,
            "lr_server": 0.0005,
            "distill_epochs": 10,
            "student_epochs": 20,
            "temperature": 5.0,
            "lambda_kd": 0.7,
            "lambda_feat": 0.1,
            "lambda_ce": 0.3,
            "normalization": {
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
        },
        "MNIST": {
            "in_channels": 3,
            "image_size": 32,
            "batch_size": 64,
            "lr_server": 0.001,
            "distill_epochs": 5,
            "student_epochs": 15,
            "temperature": 3.0,
            "lambda_kd": 0.7,
            "lambda_feat": 0.1,
            "lambda_ce": 0.3,
            "normalization": {
                "mean": (0.5, 0.5, 0.5),
                "std": (0.5, 0.5, 0.5),
            },
        },
    }

    # ─────────────────────────────────────────────────────────────────
    # Interactive configuration
    # ─────────────────────────────────────────────────────────────────
    @classmethod
    def get_federated_config(cls):
        """Interactively prompt the user for federated-learning parameters.

        Returns
        -------
        tuple of 9 ints
            (num_clients, num_clusters, clients_per_round, rounds, local_epochs,
             fmnist_distill_epochs, cifar10_distill_epochs,
             fmnist_student_epochs, cifar10_student_epochs)
        """

        def _ask_int(prompt: str, default: int) -> int:
            try:
                val = input(f"{prompt} [{default}]: ").strip()
                return int(val) if val else default
            except (ValueError, EOFError):
                return default

        num_clients = _ask_int("Number of clients", cls.num_clients)
        num_clusters = _ask_int("Number of clusters", cls.num_clusters)
        clients_per_round = _ask_int("Clients per round", cls.clients_per_round)
        rounds = _ask_int("Training rounds", cls.rounds)
        local_epochs = _ask_int("Local epochs", cls.local_epochs)
        fmnist_distill_epochs = _ask_int(
            "FashionMNIST distillation epochs",
            cls.dataset_configs["FashionMNIST"]["distill_epochs"],
        )
        cifar10_distill_epochs = _ask_int(
            "CIFAR-10 distillation epochs",
            cls.dataset_configs["CIFAR10"]["distill_epochs"],
        )
        fmnist_student_epochs = _ask_int(
            "FashionMNIST student epochs",
            cls.dataset_configs["FashionMNIST"]["student_epochs"],
        )
        cifar10_student_epochs = _ask_int(
            "CIFAR-10 student epochs",
            cls.dataset_configs["CIFAR10"]["student_epochs"],
        )

        return (
            num_clients,
            num_clusters,
            clients_per_round,
            rounds,
            local_epochs,
            fmnist_distill_epochs,
            cifar10_distill_epochs,
            fmnist_student_epochs,
            cifar10_student_epochs,
        )

    @classmethod
    def set_federated_config(
        cls,
        num_clients: int = None,
        num_clusters: int = None,
        clients_per_round: int = None,
        rounds: int = None,
        local_epochs: int = None,
        fmnist_distill_epochs: int = None,
        cifar10_distill_epochs: int = None,
        fmnist_student_epochs: int = None,
        cifar10_student_epochs: int = None,
        alpha_dirichlet: float = None,
    ):
        """Programmatically update federated-learning parameters."""
        if num_clients is not None:
            cls.num_clients = num_clients
        if num_clusters is not None:
            cls.num_clusters = num_clusters
        if clients_per_round is not None:
            cls.clients_per_round = clients_per_round
        if rounds is not None:
            cls.rounds = rounds
        if local_epochs is not None:
            cls.local_epochs = local_epochs
        if alpha_dirichlet is not None:
            cls.alpha_dirichlet = alpha_dirichlet

        # Update dataset-specific configs
        if fmnist_distill_epochs is not None:
            cls.dataset_configs["FashionMNIST"]["distill_epochs"] = fmnist_distill_epochs
        if cifar10_distill_epochs is not None:
            cls.dataset_configs["CIFAR10"]["distill_epochs"] = cifar10_distill_epochs
        if fmnist_student_epochs is not None:
            cls.dataset_configs["FashionMNIST"]["student_epochs"] = fmnist_student_epochs
        if cifar10_student_epochs is not None:
            cls.dataset_configs["CIFAR10"]["student_epochs"] = cifar10_student_epochs

        print(f"[CFG] Updated: {cls.num_clients} clients, {cls.num_clusters} clusters, "
              f"{cls.clients_per_round} clients/round, {cls.rounds} rounds, "
              f"{cls.local_epochs} local epochs")
