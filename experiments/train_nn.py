from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

# Allow this script to be run directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from gomoku.neural_network import GomokuPolicyValueNet


class GomokuSelfPlayDataset(Dataset):
    """
    PyTorch dataset for Gomoku self-play samples.

    Expected arrays:
        states:
            Shape [num_samples, 3, board_size, board_size]

        policy_targets:
            Shape [num_samples, board_size * board_size]

        value_targets:
            Shape [num_samples, 1]
    """

    def __init__(self, dataset_path: Path) -> None:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        data = np.load(dataset_path, allow_pickle=False)

        self.states = torch.tensor(data["states"], dtype=torch.float32)
        self.policy_targets = torch.tensor(data["policy_targets"], dtype=torch.float32)
        self.value_targets = torch.tensor(data["value_targets"], dtype=torch.float32)

        if self.states.ndim != 4:
            raise ValueError("states must have shape [N, 3, board_size, board_size].")

        if self.states.shape[1] != 3:
            raise ValueError("states must have exactly 3 channels.")

        if self.policy_targets.ndim != 2:
            raise ValueError("policy_targets must have shape [N, board_size * board_size].")

        if self.value_targets.ndim != 2 or self.value_targets.shape[1] != 1:
            raise ValueError("value_targets must have shape [N, 1].")

        if not (
            len(self.states)
            == len(self.policy_targets)
            == len(self.value_targets)
        ):
            raise ValueError("states, policy_targets, and value_targets must have same length.")

        self.board_size = self.states.shape[2]

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return (
            self.states[index],
            self.policy_targets[index],
            self.value_targets[index],
        )


def load_metadata(dataset_path: Path) -> dict[str, Any]:
    """
    Load metadata from the npz dataset if available.
    """
    data = np.load(dataset_path, allow_pickle=False)

    if "metadata" not in data:
        return {}

    raw_metadata = str(data["metadata"])

    try:
        return json.loads(raw_metadata)
    except json.JSONDecodeError:
        return {}


def compute_policy_loss(
    policy_logits: torch.Tensor,
    policy_targets: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss for one-hot policy targets.
    """
    target_actions = torch.argmax(policy_targets, dim=1)
    return nn.functional.cross_entropy(policy_logits, target_actions)


def train_one_epoch(
    model: GomokuPolicyValueNet,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    value_loss_weight: float,
) -> dict[str, float]:
    """
    Train the model for one epoch.
    """
    model.train()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_samples = 0

    value_loss_fn = nn.MSELoss()

    for states, policy_targets, value_targets in data_loader:
        states = states.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)

        optimizer.zero_grad()

        policy_logits, values = model(states)

        policy_loss = compute_policy_loss(policy_logits, policy_targets)
        value_loss = value_loss_fn(values, value_targets)

        loss = policy_loss + value_loss_weight * value_loss

        loss.backward()
        optimizer.step()

        batch_size = states.shape[0]
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_policy_loss += policy_loss.item() * batch_size
        total_value_loss += value_loss.item() * batch_size

    return {
        "loss": total_loss / total_samples,
        "policy_loss": total_policy_loss / total_samples,
        "value_loss": total_value_loss / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: GomokuPolicyValueNet,
    data_loader: DataLoader,
    device: torch.device,
    value_loss_weight: float,
) -> dict[str, float]:
    """
    Evaluate the model on validation data.
    """
    model.eval()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_policy_accuracy = 0.0
    total_samples = 0

    value_loss_fn = nn.MSELoss()

    for states, policy_targets, value_targets in data_loader:
        states = states.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)

        policy_logits, values = model(states)

        policy_loss = compute_policy_loss(policy_logits, policy_targets)
        value_loss = value_loss_fn(values, value_targets)
        loss = policy_loss + value_loss_weight * value_loss

        predicted_actions = torch.argmax(policy_logits, dim=1)
        target_actions = torch.argmax(policy_targets, dim=1)
        policy_accuracy = (predicted_actions == target_actions).float().mean()

        batch_size = states.shape[0]
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_policy_loss += policy_loss.item() * batch_size
        total_value_loss += value_loss.item() * batch_size
        total_policy_accuracy += policy_accuracy.item() * batch_size

    return {
        "loss": total_loss / total_samples,
        "policy_loss": total_policy_loss / total_samples,
        "value_loss": total_value_loss / total_samples,
        "policy_accuracy": total_policy_accuracy / total_samples,
    }


def save_checkpoint(
    model: GomokuPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    output_path: Path,
    epoch: int,
    board_size: int,
    metadata: dict[str, Any],
) -> None:
    """
    Save model checkpoint.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "board_size": board_size,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
        },
        output_path,
    )

    print(f"Saved checkpoint to: {output_path}")


def train_model(
    dataset_path: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    value_loss_weight: float,
    validation_split: float,
    seed: int,
    device_name: str,
) -> None:
    """
    Train a Gomoku policy-value network from a self-play dataset.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if not (0.0 <= validation_split < 1.0):
        raise ValueError("validation_split must be in [0.0, 1.0).")

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = GomokuSelfPlayDataset(dataset_path)
    metadata = load_metadata(dataset_path)

    board_size = dataset.board_size

    validation_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - validation_size

    if train_size <= 0:
        raise ValueError("Training set is empty. Reduce validation_split.")

    generator = torch.Generator().manual_seed(seed)

    if validation_size > 0:
        train_dataset, validation_dataset = random_split(
            dataset,
            [train_size, validation_size],
            generator=generator,
        )
    else:
        train_dataset = dataset
        validation_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"Using device: {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Samples: {len(dataset)}")
    print(f"Train samples: {train_size}")
    print(f"Validation samples: {validation_size}")
    print(f"Board size: {board_size}")

    model = GomokuPolicyValueNet(board_size=board_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_metadata = {
        "dataset_path": str(dataset_path),
        "dataset_metadata": metadata,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "value_loss_weight": value_loss_weight,
        "validation_split": validation_split,
        "seed": seed,
        "device": str(device),
    }

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            value_loss_weight=value_loss_weight,
        )

        message = (
            f"Epoch {epoch}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"policy={train_metrics['policy_loss']:.4f} | "
            f"value={train_metrics['value_loss']:.4f}"
        )

        if validation_loader is not None:
            val_metrics = evaluate(
                model=model,
                data_loader=validation_loader,
                device=device,
                value_loss_weight=value_loss_weight,
            )

            message += (
                f" | val loss={val_metrics['loss']:.4f} | "
                f"val policy={val_metrics['policy_loss']:.4f} | "
                f"val value={val_metrics['value_loss']:.4f} | "
                f"val acc={val_metrics['policy_accuracy']:.4f}"
            )

        print(message)

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        output_path=output_path,
        epoch=epochs,
        board_size=board_size,
        metadata=training_metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Gomoku policy-value neural network."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to self-play .npz dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/gomoku_policy_value_net.pt"),
        help="Path to save trained checkpoint.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=1.0,
        help="Weight for value loss.",
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of samples used for validation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_model(
        dataset_path=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        value_loss_weight=args.value_loss_weight,
        validation_split=args.validation_split,
        seed=args.seed,
        device_name=args.device,
    )