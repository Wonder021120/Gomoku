"""
Train the Gomoku policy-value neural network.

This version supports both:

1. Old one-hot policy targets:
   policy_target[action] = 1

2. AlphaZero-style soft policy targets:
   policy_target = MCTS visit-count distribution

It also supports the 9x9 paper-style NN-MCTS reproduction path:

- model_variant="paper_9x9"
- 4-channel states
- 32 -> 64 -> 128 policy-value network
- optional rotation/reflection symmetry augmentation
- policy entropy metric
- checkpoint model_config metadata for reliable loading by NNMCTSAgent
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from gomoku.neural_network import GomokuPolicyValueNet


class GomokuSelfPlayDataset(Dataset):
    """PyTorch dataset for Gomoku self-play samples."""

    def __init__(
        self,
        states: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
    ) -> None:
        if states.ndim != 4:
            raise ValueError(
                f"states must have shape [N, C, H, W], got {states.shape}"
            )

        if policy_targets.ndim != 2:
            raise ValueError(
                "policy_targets must have shape [N, board_size * board_size], "
                f"got {policy_targets.shape}"
            )

        if value_targets.ndim == 1:
            value_targets = value_targets.reshape(-1, 1)

        if value_targets.ndim != 2 or value_targets.shape[1] != 1:
            raise ValueError(
                f"value_targets must have shape [N, 1], got {value_targets.shape}"
            )

        sample_count = states.shape[0]

        if policy_targets.shape[0] != sample_count:
            raise ValueError("states and policy_targets have different sample counts.")

        if value_targets.shape[0] != sample_count:
            raise ValueError("states and value_targets have different sample counts.")

        self.states = torch.from_numpy(states.astype(np.float32))
        self.policy_targets = torch.from_numpy(policy_targets.astype(np.float32))
        self.value_targets = torch.from_numpy(value_targets.astype(np.float32))

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int):
        return (
            self.states[index],
            self.policy_targets[index],
            self.value_targets[index],
        )


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    """Resolve device string."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device)


def load_npz_metadata(npz_file) -> Dict:
    """Load metadata from a self-play npz file if present."""

    if "metadata" not in npz_file.files:
        return {}

    raw_metadata = npz_file["metadata"]

    try:
        if raw_metadata.shape == ():
            metadata_text = str(raw_metadata.item())
        else:
            metadata_text = str(raw_metadata)
        return json.loads(metadata_text)
    except Exception:
        return {"raw_metadata": str(raw_metadata)}


def normalize_policy_targets(policy_targets: np.ndarray) -> np.ndarray:
    """
    Normalize policy target rows.

    This is safe for both one-hot and soft policy distributions.
    Invalid rows are replaced with a uniform distribution.
    """

    policy_targets = policy_targets.astype(np.float32)
    row_sums = policy_targets.sum(axis=1, keepdims=True)

    invalid_rows = (
        ~np.isfinite(row_sums).reshape(-1)
        | (row_sums.reshape(-1) <= 0.0)
    )

    if invalid_rows.any():
        action_count = policy_targets.shape[1]
        policy_targets[invalid_rows] = 1.0 / action_count
        row_sums = policy_targets.sum(axis=1, keepdims=True)

    policy_targets = policy_targets / row_sums
    policy_targets = np.nan_to_num(policy_targets, nan=0.0, posinf=0.0, neginf=0.0)

    return policy_targets.astype(np.float32)


def transform_state_and_policy(
    state: np.ndarray,
    policy: np.ndarray,
    board_size: int,
    rotation_k: int,
    flip_horizontal: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the same board symmetry to a state tensor and a policy vector.

    Transform order:
        1. Rotate by rotation_k * 90 degrees.
        2. Optionally flip horizontally.

    This produces the 8 standard Gomoku symmetries:
        rot0, rot90, rot180, rot270,
        flip + rot0, flip + rot90, flip + rot180, flip + rot270.
    """

    if state.shape[-2:] != (board_size, board_size):
        raise ValueError("state board shape does not match board_size.")

    if policy.shape[0] != board_size * board_size:
        raise ValueError("policy length does not match board_size.")

    transformed_state = np.rot90(state, k=rotation_k, axes=(1, 2))
    policy_board = policy.reshape(board_size, board_size)
    transformed_policy_board = np.rot90(policy_board, k=rotation_k, axes=(0, 1))

    if flip_horizontal:
        transformed_state = np.flip(transformed_state, axis=2)
        transformed_policy_board = np.fliplr(transformed_policy_board)

    transformed_state = np.ascontiguousarray(transformed_state, dtype=np.float32)
    transformed_policy = np.ascontiguousarray(
        transformed_policy_board.reshape(-1),
        dtype=np.float32,
    )

    return transformed_state, transformed_policy


def augment_with_symmetries(
    states: np.ndarray,
    policy_targets: np.ndarray,
    value_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand a dataset by all 8 board symmetries.

    Value targets do not change under rotation/reflection.
    """

    if states.ndim != 4:
        raise ValueError("states must have shape [N, C, H, W].")

    board_size = int(states.shape[-1])

    if states.shape[-2] != board_size:
        raise ValueError("states must have square board dimensions.")

    augmented_states = []
    augmented_policies = []
    augmented_values = []

    for state, policy, value in zip(states, policy_targets, value_targets):
        for flip_horizontal in (False, True):
            for rotation_k in range(4):
                transformed_state, transformed_policy = transform_state_and_policy(
                    state=state,
                    policy=policy,
                    board_size=board_size,
                    rotation_k=rotation_k,
                    flip_horizontal=flip_horizontal,
                )
                augmented_states.append(transformed_state)
                augmented_policies.append(transformed_policy)
                augmented_values.append(value)

    states_out = np.stack(augmented_states, axis=0).astype(np.float32)
    policies_out = normalize_policy_targets(
        np.stack(augmented_policies, axis=0).astype(np.float32)
    )
    values_out = np.asarray(augmented_values, dtype=np.float32)

    if values_out.ndim == 1:
        values_out = values_out.reshape(-1, 1)

    return states_out, policies_out, values_out


def load_dataset(
    data_path: Path,
    augment_symmetries: bool = False,
) -> Tuple[GomokuSelfPlayDataset, Dict, int, int]:
    """Load self-play data from npz."""

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    npz_file = np.load(data_path, allow_pickle=True)

    required_keys = ["states", "policy_targets", "value_targets"]
    missing_keys = [key for key in required_keys if key not in npz_file.files]

    if missing_keys:
        raise KeyError(f"Dataset is missing keys: {missing_keys}")

    states = npz_file["states"].astype(np.float32)
    policy_targets = normalize_policy_targets(npz_file["policy_targets"])
    value_targets = npz_file["value_targets"].astype(np.float32)

    board_size = int(states.shape[-1])
    input_channels = int(states.shape[1])
    expected_action_count = board_size * board_size

    if policy_targets.shape[1] != expected_action_count:
        raise ValueError(
            f"policy_targets second dimension should be {expected_action_count}, "
            f"got {policy_targets.shape[1]}"
        )

    original_sample_count = int(states.shape[0])

    if augment_symmetries:
        states, policy_targets, value_targets = augment_with_symmetries(
            states=states,
            policy_targets=policy_targets,
            value_targets=value_targets,
        )

    metadata = load_npz_metadata(npz_file)

    dataset = GomokuSelfPlayDataset(
        states=states,
        policy_targets=policy_targets,
        value_targets=value_targets,
    )

    row_sums = policy_targets.sum(axis=1)
    nonzero_counts = (policy_targets > 1e-8).sum(axis=1)
    non_one_hot_count = int((nonzero_counts > 1).sum())

    print("Dataset loaded:")
    print(f"  path: {data_path}")
    print(f"  original samples: {original_sample_count}")
    print(f"  samples after augmentation: {len(dataset)}")
    print(f"  states shape: {states.shape}")
    print(f"  input_channels: {input_channels}")
    print(f"  policy_targets shape: {policy_targets.shape}")
    print(f"  value_targets shape: {value_targets.shape}")
    print(f"  board_size: {board_size}")
    print(f"  augment_symmetries: {augment_symmetries}")
    print(f"  policy row sum min: {row_sums.min():.6f}")
    print(f"  policy row sum max: {row_sums.max():.6f}")
    print(f"  avg policy nonzero count: {float(nonzero_counts.mean()):.2f}")
    print(f"  non-one-hot policy targets: {non_one_hot_count}/{len(nonzero_counts)}")

    if metadata:
        print("  metadata found: yes")
        if "resolved_policy_target_mode" in metadata:
            print(
                "  resolved_policy_target_mode: "
                f"{metadata['resolved_policy_target_mode']}"
            )
        if "temperature" in metadata:
            print(f"  move temperature: {metadata['temperature']}")
        if "policy_temperature" in metadata:
            print(f"  policy temperature: {metadata['policy_temperature']}")
    else:
        print("  metadata found: no")

    print()

    return dataset, metadata, board_size, input_channels


def split_dataset(
    dataset: GomokuSelfPlayDataset,
    validation_split: float,
    seed: int,
):
    """Split dataset into train and validation sets."""

    if not 0.0 <= validation_split < 1.0:
        raise ValueError("--validation-split must be in [0, 1).")

    sample_count = len(dataset)

    if sample_count <= 1 or validation_split == 0.0:
        return dataset, None

    val_count = int(round(sample_count * validation_split))
    val_count = max(1, val_count)
    val_count = min(val_count, sample_count - 1)
    train_count = sample_count - val_count

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_count, val_count],
        generator=generator,
    )

    return train_dataset, val_dataset


def get_model_outputs(model: nn.Module, states: torch.Tensor):
    """Return policy logits and value predictions from the model."""

    output = model(states)

    if hasattr(output, "policy_logits"):
        policy_logits = output.policy_logits
        value = output.value
    elif isinstance(output, tuple) and len(output) == 2:
        policy_logits, value = output
    else:
        raise TypeError("Unexpected model output format.")

    return policy_logits, value


def soft_policy_cross_entropy(
    policy_logits: torch.Tensor,
    policy_targets: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss for soft policy targets.

    Works for:
    - one-hot targets
    - soft MCTS visit distributions
    """

    log_probs = torch.log_softmax(policy_logits, dim=1)
    return -(policy_targets * log_probs).sum(dim=1).mean()


def policy_entropy(policy_logits: torch.Tensor) -> torch.Tensor:
    """
    Mean entropy of the model policy distribution.

    Higher entropy means the policy is more diffuse.
    Lower entropy means the model is more confident/concentrated.
    """

    probabilities = torch.softmax(policy_logits, dim=1)
    log_probs = torch.log_softmax(policy_logits, dim=1)
    entropy = -(probabilities * log_probs).sum(dim=1)
    return entropy.mean()


def target_policy_entropy(policy_targets: torch.Tensor) -> torch.Tensor:
    """Mean entropy of target policy distributions."""

    eps = 1e-12
    safe_targets = torch.clamp(policy_targets, min=eps)
    entropy = -(policy_targets * torch.log(safe_targets)).sum(dim=1)
    return entropy.mean()


def policy_argmax_accuracy(
    policy_logits: torch.Tensor,
    policy_targets: torch.Tensor,
) -> Tuple[int, int]:
    """
    Argmax policy accuracy.

    This is only a diagnostic metric. For soft MCTS targets, accuracy is less
    important than the soft policy loss.
    """

    predicted_actions = torch.argmax(policy_logits, dim=1)
    target_actions = torch.argmax(policy_targets, dim=1)

    correct = int((predicted_actions == target_actions).sum().item())
    total = int(policy_targets.shape[0])

    return correct, total


def create_model(
    board_size: int,
    input_channels: int,
    model_variant: str,
) -> Tuple[nn.Module, Dict]:
    """Create a model and return model_config metadata."""

    variant = str(model_variant).strip().lower().replace("-", "_").replace(" ", "_")

    if variant in ("legacy", "old", "default"):
        model = GomokuPolicyValueNet(
            board_size=board_size,
            input_channels=input_channels,
        )
        model_config = {
            "model_variant": "legacy",
            "board_size": board_size,
            "input_channels": input_channels,
            "conv_channels": list(getattr(model, "conv_channels", (64, 64, 64))),
            "policy_head_channels": getattr(model, "policy_head_channels", 2),
            "value_head_channels": getattr(model, "value_head_channels", 1),
            "value_hidden_size": getattr(model, "value_hidden_size", 64),
        }
        return model, model_config

    if variant in ("paper", "paper_9x9", "repro", "reproduction", "svn"):
        if board_size != 9:
            raise ValueError("--model-variant paper_9x9 requires board_size=9.")
        if input_channels != 4:
            raise ValueError(
                "--model-variant paper_9x9 requires 4-channel states. "
                "Regenerate self-play data with paper_9x9 NN-MCTS encoding."
            )

        model = GomokuPolicyValueNet.create_paper_9x9()
        model_config = {
            "model_variant": "paper_9x9",
            "board_size": 9,
            "input_channels": 4,
            "conv_channels": [32, 64, 128],
            "policy_head_channels": 4,
            "value_head_channels": 2,
            "value_hidden_size": 64,
        }
        return model, model_config

    raise ValueError(
        f"Unknown model variant: {model_variant!r}. "
        "Valid values: legacy, paper_9x9."
    )


def load_checkpoint_if_requested(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Optional[Path],
    device: torch.device,
) -> int:
    """
    Load model and optimizer checkpoint if requested.

    Returns:
        start_epoch
    """

    if checkpoint_path is None:
        return 1

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    previous_epoch = int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0
    start_epoch = previous_epoch + 1

    print(f"Resumed from checkpoint: {checkpoint_path}")
    print(f"Starting from epoch: {start_epoch}")
    print()

    return start_epoch


def run_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    value_loss_weight: float,
) -> Dict[str, float]:
    """
    Run one train or validation epoch.

    If optimizer is None, runs evaluation mode without gradient updates.
    """

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    value_loss_fn = nn.MSELoss()

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    policy_entropy_sum = 0.0
    target_entropy_sum = 0.0
    correct_actions = 0
    total_samples = 0

    for states, policy_targets, value_targets in data_loader:
        states = states.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            policy_logits, predicted_values = get_model_outputs(model, states)

            policy_loss = soft_policy_cross_entropy(
                policy_logits=policy_logits,
                policy_targets=policy_targets,
            )

            value_loss = value_loss_fn(predicted_values, value_targets)
            total_loss = policy_loss + value_loss_weight * value_loss

            if is_training:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = states.shape[0]

        total_loss_sum += float(total_loss.item()) * batch_size
        policy_loss_sum += float(policy_loss.item()) * batch_size
        value_loss_sum += float(value_loss.item()) * batch_size

        with torch.no_grad():
            model_entropy = policy_entropy(policy_logits.detach())
            target_entropy = target_policy_entropy(policy_targets.detach())

        policy_entropy_sum += float(model_entropy.item()) * batch_size
        target_entropy_sum += float(target_entropy.item()) * batch_size

        correct, total = policy_argmax_accuracy(
            policy_logits=policy_logits.detach(),
            policy_targets=policy_targets.detach(),
        )

        correct_actions += correct
        total_samples += total

    if total_samples == 0:
        raise ValueError("DataLoader produced no samples.")

    return {
        "loss": total_loss_sum / total_samples,
        "policy_loss": policy_loss_sum / total_samples,
        "value_loss": value_loss_sum / total_samples,
        "policy_entropy": policy_entropy_sum / total_samples,
        "target_policy_entropy": target_entropy_sum / total_samples,
        "policy_argmax_accuracy": correct_actions / total_samples,
    }


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    board_size: int,
    input_channels: int,
    model_config: Dict,
    dataset_metadata: Dict,
    training_config: Dict,
    train_metrics: Dict[str, float],
    val_metrics: Optional[Dict[str, float]],
) -> None:
    """Save training checkpoint."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "board_size": board_size,
        "input_channels": input_channels,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dataset_metadata": dataset_metadata,
        "training_config": training_config,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    torch.save(checkpoint, output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train Gomoku policy-value neural network."
    )

    parser.add_argument(
        "--data",
        "--dataset",
        "--input",
        "--input-path",
        dest="data_path",
        type=Path,
        required=True,
        help="Path to self-play npz dataset.",
    )

    parser.add_argument(
        "--output",
        "--checkpoint-output",
        "--checkpoint-path",
        dest="output_path",
        type=Path,
        default=Path("checkpoints/gomoku_policy_value_net.pt"),
        help="Path where the final checkpoint will be saved.",
    )

    parser.add_argument(
        "--best-output",
        type=Path,
        default=None,
        help="Optional path where the best validation checkpoint will be saved.",
    )

    parser.add_argument(
        "--model-variant",
        type=str,
        default="legacy",
        choices=["legacy", "paper_9x9"],
        help="Network architecture to train. Use paper_9x9 for the 9x9 reproduction model.",
    )

    parser.add_argument(
        "--augment-symmetries",
        action="store_true",
        help="Apply all 8 rotation/reflection board symmetries to the dataset.",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to resume training from.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")

    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative.")

    if args.value_loss_weight < 0.0:
        raise ValueError("--value-loss-weight must be non-negative.")

    if not 0.0 <= args.validation_split < 1.0:
        raise ValueError("--validation-split must be in [0, 1).")

    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")


def print_epoch_metrics(
    epoch: int,
    total_epochs: int,
    train_metrics: Dict[str, float],
    val_metrics: Optional[Dict[str, float]],
) -> None:
    """Print compact training metrics."""

    if val_metrics is not None:
        print(
            f"Epoch {epoch:03d}/{total_epochs:03d} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"policy={train_metrics['policy_loss']:.4f} "
            f"value={train_metrics['value_loss']:.4f} "
            f"entropy={train_metrics['policy_entropy']:.4f} "
            f"target_entropy={train_metrics['target_policy_entropy']:.4f} "
            f"acc={train_metrics['policy_argmax_accuracy']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"policy={val_metrics['policy_loss']:.4f} "
            f"value={val_metrics['value_loss']:.4f} "
            f"entropy={val_metrics['policy_entropy']:.4f} "
            f"target_entropy={val_metrics['target_policy_entropy']:.4f} "
            f"acc={val_metrics['policy_argmax_accuracy']:.4f}"
        )
    else:
        print(
            f"Epoch {epoch:03d}/{total_epochs:03d} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"policy={train_metrics['policy_loss']:.4f} "
            f"value={train_metrics['value_loss']:.4f} "
            f"entropy={train_metrics['policy_entropy']:.4f} "
            f"target_entropy={train_metrics['target_policy_entropy']:.4f} "
            f"acc={train_metrics['policy_argmax_accuracy']:.4f}"
        )


def main() -> None:
    """Train the policy-value network."""

    args = parse_args()
    validate_args(args)

    set_random_seed(args.seed)
    device = resolve_device(args.device)

    dataset, dataset_metadata, board_size, input_channels = load_dataset(
        data_path=args.data_path,
        augment_symmetries=args.augment_symmetries,
    )

    train_dataset, val_dataset = split_dataset(
        dataset=dataset,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = None

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    model, model_config = create_model(
        board_size=board_size,
        input_channels=input_channels,
        model_variant=args.model_variant,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch = load_checkpoint_if_requested(
        model=model,
        optimizer=optimizer,
        checkpoint_path=args.resume_checkpoint,
        device=device,
    )

    training_config = {
        "data_path": str(args.data_path),
        "output_path": str(args.output_path),
        "best_output": str(args.best_output) if args.best_output else None,
        "model_variant": args.model_variant,
        "augment_symmetries": args.augment_symmetries,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "value_loss_weight": args.value_loss_weight,
        "validation_split": args.validation_split,
        "seed": args.seed,
        "device": str(device),
        "num_workers": args.num_workers,
        "resume_checkpoint": str(args.resume_checkpoint)
        if args.resume_checkpoint
        else None,
        "policy_loss": "soft_cross_entropy",
        "metrics": ["policy_argmax_accuracy", "policy_entropy", "target_policy_entropy"],
    }

    print("Training configuration:")
    print(f"  device: {device}")
    print(f"  train samples: {len(train_dataset)}")
    print(f"  validation samples: {len(val_dataset) if val_dataset is not None else 0}")
    print(f"  board_size: {board_size}")
    print(f"  input_channels: {input_channels}")
    print(f"  model_variant: {args.model_variant}")
    print(f"  model_config: {model_config}")
    print(f"  augment_symmetries: {args.augment_symmetries}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  value_loss_weight: {args.value_loss_weight}")
    print(f"  policy_loss: soft_cross_entropy")
    print(f"  output_path: {args.output_path}")

    if args.best_output is not None:
        print(f"  best_output: {args.best_output}")

    print()

    best_val_loss = float("inf")
    best_epoch = None
    train_metrics = {}

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            value_loss_weight=args.value_loss_weight,
        )

        val_metrics = None

        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_one_epoch(
                    model=model,
                    data_loader=val_loader,
                    optimizer=None,
                    device=device,
                    value_loss_weight=args.value_loss_weight,
                )

        print_epoch_metrics(
            epoch=epoch,
            total_epochs=args.epochs,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        if (
            val_metrics is not None
            and args.best_output is not None
            and val_metrics["loss"] < best_val_loss
        ):
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch

            save_checkpoint(
                output_path=args.best_output,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                board_size=board_size,
                input_channels=input_channels,
                model_config=model_config,
                dataset_metadata=dataset_metadata,
                training_config=training_config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

            print(f"  Saved best checkpoint: {args.best_output}")

    final_val_metrics = None

    if val_loader is not None:
        with torch.no_grad():
            final_val_metrics = run_one_epoch(
                model=model,
                data_loader=val_loader,
                optimizer=None,
                device=device,
                value_loss_weight=args.value_loss_weight,
            )

    save_checkpoint(
        output_path=args.output_path,
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        board_size=board_size,
        input_channels=input_channels,
        model_config=model_config,
        dataset_metadata=dataset_metadata,
        training_config=training_config,
        train_metrics=train_metrics,
        val_metrics=final_val_metrics,
    )

    print()
    print(f"Saved final checkpoint: {args.output_path}")

    if best_epoch is not None:
        print(
            f"Best validation checkpoint: {args.best_output} "
            f"at epoch {best_epoch} with val loss {best_val_loss:.4f}"
        )


if __name__ == "__main__":
    main()
