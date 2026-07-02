"""
Generate self-play data for training the Gomoku policy-value network.

This script supports Random, Greedy, and NN-MCTS self-play data generation.

For NN-MCTS, this version supports AlphaZero-style policy targets:
instead of storing only the final selected move as a one-hot vector, it can store
the MCTS visit-count policy distribution returned by NNMCTSAgent.

This is still a lightweight implementation, but it is closer to AlphaZero-style
training than the previous one-hot-only pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

from gomoku.agents import GreedyAgent, RandomAgent
from gomoku.board import Board
from gomoku.game import Game, GameStatus
from gomoku.neural_network import encode_board_state, move_to_action
from gomoku.nn_mcts_agent import NNMCTSAgent


Move = Tuple[int, int]


@dataclass
class SelfPlaySample:
    """One training sample collected before a move is played."""

    state: np.ndarray
    policy_target: np.ndarray
    current_player: int
    value_target: float = 0.0


def create_agent(
    agent_name: str,
    board_size: int,
    seed: int,
    nn_mcts_simulations: int,
    nn_mcts_checkpoint: Optional[str],
    nn_mcts_exploration_weight: float,
    nn_mcts_candidate_radius: int,
    nn_mcts_device: str,
):
    """Create an agent for self-play data generation."""

    if agent_name == "random":
        try:
            return RandomAgent(seed=seed)
        except TypeError:
            return RandomAgent()

    if agent_name == "greedy":
        try:
            return GreedyAgent(seed=seed)
        except TypeError:
            return GreedyAgent()

    if agent_name == "nn_mcts":
        return NNMCTSAgent(
            board_size=board_size,
            simulations=nn_mcts_simulations,
            checkpoint_path=nn_mcts_checkpoint,
            exploration_weight=nn_mcts_exploration_weight,
            candidate_radius=nn_mcts_candidate_radius,
            device=nn_mcts_device,
            seed=seed,
        )

    raise ValueError(f"Unsupported agent: {agent_name}")


def build_one_hot_policy_target(board_size: int, move: Move) -> np.ndarray:
    """Build a one-hot policy target for a selected move."""

    policy_target = np.zeros(board_size * board_size, dtype=np.float32)
    action = move_to_action(move, board_size)
    policy_target[action] = 1.0
    return policy_target


def normalize_policy_target(
    policy_target: np.ndarray,
    board_size: int,
    fallback_move: Move,
) -> np.ndarray:
    """
    Validate and normalize a policy target.

    If the policy is invalid, fall back to one-hot encoding of the selected move.
    """

    expected_shape = (board_size * board_size,)
    policy = np.asarray(policy_target, dtype=np.float32)

    if policy.shape != expected_shape:
        return build_one_hot_policy_target(board_size, fallback_move)

    total = float(policy.sum())

    if total <= 0.0 or not np.isfinite(total):
        return build_one_hot_policy_target(board_size, fallback_move)

    policy = policy / total

    if not np.isfinite(policy).all():
        return build_one_hot_policy_target(board_size, fallback_move)

    return policy.astype(np.float32)


def get_nearby_candidate_moves(game: Game, radius: int) -> List[Move]:
    """
    Return legal moves near existing stones.

    This is kept for legacy one-hot opening exploration. For the new
    AlphaZero-style NN-MCTS mode, exploration should normally come from
    temperature sampling over the MCTS visit distribution instead.
    """

    legal_moves = game.get_legal_moves()

    if not legal_moves:
        return []

    if not game.board.move_history:
        centre = game.board.size // 2
        centre_move = (centre, centre)

        if centre_move in legal_moves:
            return [centre_move]

        return legal_moves

    legal_set = set(legal_moves)
    candidate_set = set()

    for (row, col), _player in game.board.move_history:
        for candidate_row in range(row - radius, row + radius + 1):
            for candidate_col in range(col - radius, col + radius + 1):
                candidate_move = (candidate_row, candidate_col)

                if candidate_move in legal_set:
                    candidate_set.add(candidate_move)

    if not candidate_set:
        return legal_moves

    return sorted(candidate_set)


def resolve_policy_target_mode(agent_name: str, requested_mode: str) -> str:
    """
    Resolve policy target mode.

    auto:
        nn_mcts -> visit_distribution
        random/greedy -> one_hot
    """

    if requested_mode == "auto":
        if agent_name == "nn_mcts":
            return "visit_distribution"
        return "one_hot"

    if requested_mode == "visit_distribution" and agent_name != "nn_mcts":
        raise ValueError(
            "--policy-target-mode visit_distribution is only supported for nn_mcts."
        )

    return requested_mode


def choose_move_and_policy_target(
    game: Game,
    agent,
    agent_name: str,
    rng: random.Random,
    policy_target_mode: str,
    temperature_moves: int,
    temperature: float,
    policy_temperature: float,
    board_size: int,
    opening_exploration_moves: int,
    opening_exploration_rate: float,
    opening_exploration_radius: int,
) -> Tuple[Move, np.ndarray, str]:
    """
    Choose a move and return the corresponding policy target.

    For NN-MCTS with visit_distribution:
        - Use select_move_with_policy().
        - During early moves, temperature sampling can be enabled.
        - Save the returned MCTS visit distribution as the policy target.

    For one_hot:
        - Optionally use legacy local opening exploration.
        - Save the final selected move as one-hot.
    """

    move_count = len(game.board.move_history)

    if agent_name == "nn_mcts" and policy_target_mode == "visit_distribution":
        move_temperature = temperature if move_count < temperature_moves else 0.0
        move, policy_target = agent.select_move_with_policy(
            game=game,
            move_temperature=move_temperature,
            policy_temperature=policy_temperature,
            rng=rng,
        )

        policy_target = normalize_policy_target(
            policy_target=policy_target,
            board_size=board_size,
            fallback_move=move,
        )

        return move, policy_target, "visit_distribution"

    should_use_opening_exploration = (
        move_count > 0
        and move_count < opening_exploration_moves
        and opening_exploration_rate > 0.0
        and rng.random() < opening_exploration_rate
    )

    if should_use_opening_exploration:
        candidate_moves = get_nearby_candidate_moves(
            game=game,
            radius=opening_exploration_radius,
        )

        if candidate_moves:
            move = rng.choice(candidate_moves)
            return move, build_one_hot_policy_target(board_size, move), "one_hot_explore"

    move = agent.select_move(game)
    return move, build_one_hot_policy_target(board_size, move), "one_hot"


def get_winner_from_status(status: GameStatus) -> int:
    """Convert GameStatus into a winner value."""

    if status == GameStatus.BLACK_WIN:
        return Board.BLACK

    if status == GameStatus.WHITE_WIN:
        return Board.WHITE

    return Board.EMPTY


def play_self_play_game(
    board_size: int,
    rule: str,
    agent_name: str,
    seed: int,
    max_moves: int,
    nn_mcts_simulations: int,
    nn_mcts_checkpoint: Optional[str],
    nn_mcts_exploration_weight: float,
    nn_mcts_candidate_radius: int,
    nn_mcts_device: str,
    policy_target_mode: str,
    temperature_moves: int,
    temperature: float,
    policy_temperature: float,
    opening_exploration_moves: int,
    opening_exploration_rate: float,
    opening_exploration_radius: int,
) -> Tuple[List[SelfPlaySample], int, GameStatus, int, List[int], int]:
    """
    Play one self-play game and return collected samples.

    Returns:
        samples
        winner
        final status
        legacy opening exploration count
        policy nonzero counts
        visit-distribution target count
    """

    rng = random.Random(seed)

    game = Game(board_size=board_size, rule_name=rule)

    black_agent = create_agent(
        agent_name=agent_name,
        board_size=board_size,
        seed=seed,
        nn_mcts_simulations=nn_mcts_simulations,
        nn_mcts_checkpoint=nn_mcts_checkpoint,
        nn_mcts_exploration_weight=nn_mcts_exploration_weight,
        nn_mcts_candidate_radius=nn_mcts_candidate_radius,
        nn_mcts_device=nn_mcts_device,
    )

    white_agent = create_agent(
        agent_name=agent_name,
        board_size=board_size,
        seed=seed + 1,
        nn_mcts_simulations=nn_mcts_simulations,
        nn_mcts_checkpoint=nn_mcts_checkpoint,
        nn_mcts_exploration_weight=nn_mcts_exploration_weight,
        nn_mcts_candidate_radius=nn_mcts_candidate_radius,
        nn_mcts_device=nn_mcts_device,
    )

    samples: List[SelfPlaySample] = []
    legacy_exploration_count = 0
    policy_nonzero_counts: List[int] = []
    visit_distribution_count = 0

    while game.status == GameStatus.ONGOING and len(game.board.move_history) < max_moves:
        current_player = game.current_player
        agent = black_agent if current_player == Board.BLACK else white_agent

        state = encode_board_state(game.board, current_player)

        move, policy_target, target_kind = choose_move_and_policy_target(
            game=game,
            agent=agent,
            agent_name=agent_name,
            rng=rng,
            policy_target_mode=policy_target_mode,
            temperature_moves=temperature_moves,
            temperature=temperature,
            policy_temperature=policy_temperature,
            board_size=board_size,
            opening_exploration_moves=opening_exploration_moves,
            opening_exploration_rate=opening_exploration_rate,
            opening_exploration_radius=opening_exploration_radius,
        )

        if target_kind == "one_hot_explore":
            legacy_exploration_count += 1

        if target_kind == "visit_distribution":
            visit_distribution_count += 1

        policy_nonzero_counts.append(int(np.count_nonzero(policy_target > 1e-8)))

        samples.append(
            SelfPlaySample(
                state=state,
                policy_target=policy_target,
                current_player=current_player,
            )
        )

        game.play_move(move)

    winner = get_winner_from_status(game.status)

    if game.status == GameStatus.ONGOING:
        winner = Board.EMPTY

    for sample in samples:
        if winner == Board.EMPTY:
            sample.value_target = 0.0
        elif sample.current_player == winner:
            sample.value_target = 1.0
        else:
            sample.value_target = -1.0

    return (
        samples,
        winner,
        game.status,
        legacy_exploration_count,
        policy_nonzero_counts,
        visit_distribution_count,
    )


def save_dataset(
    samples: List[SelfPlaySample],
    output_path: Path,
    metadata: dict,
) -> None:
    """Save self-play samples to a compressed npz file."""

    if not samples:
        raise ValueError("No self-play samples were generated.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    states = np.stack([sample.state for sample in samples]).astype(np.float32)
    policy_targets = np.stack(
        [sample.policy_target for sample in samples]
    ).astype(np.float32)
    value_targets = np.array(
        [[sample.value_target] for sample in samples],
        dtype=np.float32,
    )
    current_players = np.array(
        [[sample.current_player] for sample in samples],
        dtype=np.int8,
    )

    np.savez_compressed(
        output_path,
        states=states,
        policy_targets=policy_targets,
        value_targets=value_targets,
        current_players=current_players,
        metadata=json.dumps(metadata),
    )

    row_sums = policy_targets.sum(axis=1)
    nonzero_counts = (policy_targets > 1e-8).sum(axis=1)
    non_one_hot_count = int((nonzero_counts > 1).sum())

    print(f"Saved self-play dataset to: {output_path}")
    print(f"states shape: {states.shape}")
    print(f"policy_targets shape: {policy_targets.shape}")
    print(f"value_targets shape: {value_targets.shape}")
    print(f"current_players shape: {current_players.shape}")
    print(f"policy row sum min: {row_sums.min():.6f}")
    print(f"policy row sum max: {row_sums.max():.6f}")
    print(f"avg policy nonzero count: {float(nonzero_counts.mean()):.2f}")
    print(f"non-one-hot policy targets: {non_one_hot_count}/{len(nonzero_counts)}")


def build_default_output_path(
    output_dir: Path,
    agent: str,
    rule: str,
    board_size: int,
    games: int,
    seed: int,
    policy_target_mode: str,
    temperature_moves: int,
    temperature: float,
) -> Path:
    """Build a readable default output filename."""

    temperature_text = str(temperature).replace(".", "p")

    filename = (
        f"self_play_{agent}_{rule}_{board_size}x{board_size}"
        f"_{games}games_seed{seed}_{policy_target_mode}"
        f"_temp{temperature_text}_moves{temperature_moves}.npz"
    )

    return output_dir / filename


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate Gomoku self-play data for neural-network training."
    )

    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--board-size", type=int, default=15)

    parser.add_argument(
        "--rule",
        choices=["standard", "pro"],
        default="standard",
        help="Rule used during self-play data generation.",
    )

    parser.add_argument(
        "--agent",
        choices=["random", "greedy", "nn_mcts"],
        default="greedy",
        help="Agent used for self-play.",
    )

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-moves", type=int, default=150)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/self_play"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output npz path.",
    )

    parser.add_argument(
        "--policy-target-mode",
        choices=["auto", "one_hot", "visit_distribution"],
        default="auto",
        help=(
            "Policy target type. auto uses visit_distribution for nn_mcts "
            "and one_hot for random/greedy."
        ),
    )

    parser.add_argument(
        "--temperature-moves",
        type=int,
        default=8,
        help=(
            "Number of early moves using temperature sampling for NN-MCTS "
            "visit-distribution self-play."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature used during early NN-MCTS self-play moves.",
    )

    parser.add_argument(
    "--policy-temperature",
    type=float,
    default=1.0,
    help=(
        "Temperature used when saving NN-MCTS visit distributions as "
        "policy targets. Keep this at 1.0 for soft AlphaZero-style targets."
    ),
)

    # NN-MCTS options
    parser.add_argument("--nn-mcts-simulations", type=int, default=25)
    parser.add_argument("--nn-mcts-checkpoint", type=str, default=None)
    parser.add_argument("--nn-mcts-exploration-weight", type=float, default=1.5)
    parser.add_argument("--nn-mcts-candidate-radius", type=int, default=2)
    parser.add_argument("--nn-mcts-device", type=str, default="auto")

    # Legacy local opening exploration options for one-hot mode
    parser.add_argument(
        "--opening-exploration-moves",
        type=int,
        default=0,
        help=(
            "Legacy one-hot local opening exploration. Ignored for NN-MCTS "
            "visit_distribution mode."
        ),
    )

    parser.add_argument(
        "--opening-exploration-rate",
        type=float,
        default=0.0,
        help="Legacy one-hot local opening exploration probability.",
    )

    parser.add_argument(
        "--opening-exploration-radius",
        type=int,
        default=3,
        help="Legacy one-hot local opening exploration radius.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""

    if args.games <= 0:
        raise ValueError("--games must be positive.")

    if args.board_size <= 0:
        raise ValueError("--board-size must be positive.")

    if args.max_moves <= 0:
        raise ValueError("--max-moves must be positive.")

    if args.nn_mcts_simulations <= 0:
        raise ValueError("--nn-mcts-simulations must be positive.")

    if args.nn_mcts_candidate_radius <= 0:
        raise ValueError("--nn-mcts-candidate-radius must be positive.")

    if args.temperature_moves < 0:
        raise ValueError("--temperature-moves must be non-negative.")

    if args.policy_temperature < 0.0:
        raise ValueError("--policy-temperature must be non-negative.")

    if args.opening_exploration_moves < 0:
        raise ValueError("--opening-exploration-moves must be non-negative.")

    if not 0.0 <= args.opening_exploration_rate <= 1.0:
        raise ValueError("--opening-exploration-rate must be between 0 and 1.")

    if args.opening_exploration_radius <= 0:
        raise ValueError("--opening-exploration-radius must be positive.")


def main() -> None:
    """Generate and save self-play data."""

    args = parse_args()
    validate_args(args)

    resolved_policy_target_mode = resolve_policy_target_mode(
        agent_name=args.agent,
        requested_mode=args.policy_target_mode,
    )

    if args.agent == "nn_mcts" and args.nn_mcts_checkpoint is None:
        print(
            "Warning: NN-MCTS is running without a checkpoint. "
            "The neural network will use random initial weights."
        )

    output_path = args.output or build_default_output_path(
        output_dir=args.output_dir,
        agent=args.agent,
        rule=args.rule,
        board_size=args.board_size,
        games=args.games,
        seed=args.seed,
        policy_target_mode=resolved_policy_target_mode,
        temperature_moves=args.temperature_moves,
        temperature=args.temperature,
    )

    all_samples: List[SelfPlaySample] = []
    game_lengths: List[int] = []
    winners: List[int] = []
    legacy_exploration_counts: List[int] = []
    all_policy_nonzero_counts: List[int] = []
    visit_distribution_counts: List[int] = []

    print("Self-play configuration:")
    print(f"  games: {args.games}")
    print(f"  board_size: {args.board_size}")
    print(f"  rule: {args.rule}")
    print(f"  agent: {args.agent}")
    print(f"  max_moves: {args.max_moves}")
    print(f"  seed: {args.seed}")
    print(f"  requested_policy_target_mode: {args.policy_target_mode}")
    print(f"  resolved_policy_target_mode: {resolved_policy_target_mode}")
    print(f"  temperature_moves: {args.temperature_moves}")
    print(f"  temperature: {args.temperature}")
    print(f"  policy_temperature: {args.policy_temperature}")
    print(f"  nn_mcts_simulations: {args.nn_mcts_simulations}")
    print(f"  nn_mcts_checkpoint: {args.nn_mcts_checkpoint}")
    print(f"  nn_mcts_exploration_weight: {args.nn_mcts_exploration_weight}")
    print(f"  nn_mcts_candidate_radius: {args.nn_mcts_candidate_radius}")
    print(f"  nn_mcts_device: {args.nn_mcts_device}")
    print(f"  legacy_opening_exploration_moves: {args.opening_exploration_moves}")
    print(f"  legacy_opening_exploration_rate: {args.opening_exploration_rate}")
    print(f"  legacy_opening_exploration_radius: {args.opening_exploration_radius}")
    print()

    for game_index in tqdm(range(args.games), desc="Generating self-play games"):
        game_seed = args.seed + game_index

        (
            samples,
            winner,
            status,
            legacy_exploration_count,
            policy_nonzero_counts,
            visit_distribution_count,
        ) = play_self_play_game(
            board_size=args.board_size,
            rule=args.rule,
            agent_name=args.agent,
            seed=game_seed,
            max_moves=args.max_moves,
            nn_mcts_simulations=args.nn_mcts_simulations,
            nn_mcts_checkpoint=args.nn_mcts_checkpoint,
            nn_mcts_exploration_weight=args.nn_mcts_exploration_weight,
            nn_mcts_candidate_radius=args.nn_mcts_candidate_radius,
            nn_mcts_device=args.nn_mcts_device,
            policy_target_mode=resolved_policy_target_mode,
            temperature_moves=args.temperature_moves,
            temperature=args.temperature,
            policy_temperature=args.policy_temperature,
            opening_exploration_moves=args.opening_exploration_moves,
            opening_exploration_rate=args.opening_exploration_rate,
            opening_exploration_radius=args.opening_exploration_radius,
        )

        all_samples.extend(samples)
        game_lengths.append(len(samples))
        winners.append(winner)
        legacy_exploration_counts.append(legacy_exploration_count)
        all_policy_nonzero_counts.extend(policy_nonzero_counts)
        visit_distribution_counts.append(visit_distribution_count)

        avg_nonzero = float(np.mean(policy_nonzero_counts)) if policy_nonzero_counts else 0.0

        print(
            f"Game {game_index + 1}/{args.games}: "
            f"{len(samples)} samples, "
            f"winner={winner}, "
            f"status={status.name}, "
            f"visit_distribution_targets={visit_distribution_count}, "
            f"avg_policy_nonzero={avg_nonzero:.2f}"
        )

    metadata = {
        "games": args.games,
        "board_size": args.board_size,
        "rule": args.rule,
        "agent": args.agent,
        "seed": args.seed,
        "max_moves": args.max_moves,
        "requested_policy_target_mode": args.policy_target_mode,
        "resolved_policy_target_mode": resolved_policy_target_mode,
        "temperature_moves": args.temperature_moves,
        "temperature": args.temperature,
        "policy_temperature": args.policy_temperature,
        "nn_mcts_simulations": args.nn_mcts_simulations,
        "nn_mcts_checkpoint": args.nn_mcts_checkpoint,
        "nn_mcts_exploration_weight": args.nn_mcts_exploration_weight,
        "nn_mcts_candidate_radius": args.nn_mcts_candidate_radius,
        "nn_mcts_device": args.nn_mcts_device,
        "legacy_opening_exploration_moves": args.opening_exploration_moves,
        "legacy_opening_exploration_rate": args.opening_exploration_rate,
        "legacy_opening_exploration_radius": args.opening_exploration_radius,
        "game_lengths": game_lengths,
        "winners": winners,
        "legacy_exploration_counts": legacy_exploration_counts,
        "policy_nonzero_counts": all_policy_nonzero_counts,
        "visit_distribution_counts": visit_distribution_counts,
    }

    save_dataset(
        samples=all_samples,
        output_path=output_path,
        metadata=metadata,
    )

    unique_lengths = sorted(set(game_lengths))
    avg_policy_nonzero = (
        float(np.mean(all_policy_nonzero_counts))
        if all_policy_nonzero_counts
        else 0.0
    )

    print()
    print("Generation summary:")
    print(f"  total samples: {len(all_samples)}")
    print(f"  game lengths: {game_lengths}")
    print(f"  unique game lengths: {unique_lengths}")
    print(f"  total visit-distribution targets: {sum(visit_distribution_counts)}")
    print(f"  avg policy nonzero count: {avg_policy_nonzero:.2f}")
    print(f"  output: {output_path}")


if __name__ == "__main__":
    main()