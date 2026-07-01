"""
Generate self-play data for training the Gomoku policy-value network.

This script supports Random, Greedy, and NN-MCTS self-play data generation.

For NN-MCTS self-play, this file also supports optional opening exploration.
Opening exploration is only used during data generation, not during formal
tournament experiments.

The purpose of opening exploration is to avoid fully deterministic self-play
games where every game follows the same move sequence. Exploration is local:
it samples from legal moves near existing stones instead of sampling randomly
from the whole board.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure the project root is importable when running:
# python experiments/generate_self_play_data.py
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
        )

    raise ValueError(f"Unsupported agent: {agent_name}")


def build_policy_target(board_size: int, move: Move) -> np.ndarray:
    """
    Build a one-hot policy target.

    This project uses a simplified NN-MCTS training target:
    the final move selected during self-play is encoded as 1, and all other
    board positions are encoded as 0.

    This is different from full AlphaZero-style training, where the full MCTS
    visit-count distribution is usually stored as the policy target.
    """

    policy_target = np.zeros(board_size * board_size, dtype=np.float32)
    action = move_to_action(move, board_size)
    policy_target[action] = 1.0
    return policy_target


def get_nearby_candidate_moves(game: Game, radius: int) -> List[Move]:
    """
    Return legal moves near existing stones.

    This function is used for opening exploration. It prevents unrealistic
    full-board random moves, such as placing the second move in a far corner.

    If the board is empty, the centre move is returned when legal.
    Otherwise, legal moves within the given radius of existing stones are used.
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

    candidate_set = set()

    for (row, col), _player in game.board.move_history:
        for candidate_row in range(row - radius, row + radius + 1):
            for candidate_col in range(col - radius, col + radius + 1):
                candidate_move = (candidate_row, candidate_col)

                if candidate_move in legal_moves:
                    candidate_set.add(candidate_move)

    if not candidate_set:
        return legal_moves

    return sorted(candidate_set)


def choose_self_play_move(
    game: Game,
    agent,
    rng: random.Random,
    opening_exploration_moves: int,
    opening_exploration_rate: float,
    opening_exploration_radius: int,
) -> Tuple[Move, bool]:
    """
    Choose a move during self-play.

    Opening exploration logic:

    - move_count == 0:
      Do not explore. This keeps the first move controlled by the agent.
      For NN-MCTS, this is usually the centre move.

    - 0 < move_count < opening_exploration_moves:
      Exploration is allowed with probability opening_exploration_rate.
      If exploration happens, a legal move is sampled from nearby candidate
      moves around existing stones.

    - move_count >= opening_exploration_moves:
      Exploration stops. The move is fully selected by the agent.
    """

    move_count = len(game.board.move_history)

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
            return rng.choice(candidate_moves), True

    return agent.select_move(game), False


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
    opening_exploration_moves: int,
    opening_exploration_rate: float,
    opening_exploration_radius: int,
) -> Tuple[List[SelfPlaySample], int, GameStatus, int]:
    """
    Play one self-play game and return collected samples.

    Returns:
        samples:
            Training samples from this game.
        winner:
            Board.BLACK, Board.WHITE, or Board.EMPTY.
        status:
            Final GameStatus.
        exploration_count:
            Number of opening exploration moves used in this game.
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
    exploration_count = 0

    while game.status == GameStatus.ONGOING and len(game.board.move_history) < max_moves:
        current_player = game.current_player
        agent = black_agent if current_player == Board.BLACK else white_agent

        state = encode_board_state(game.board, current_player)

        move, used_opening_exploration = choose_self_play_move(
            game=game,
            agent=agent,
            rng=rng,
            opening_exploration_moves=opening_exploration_moves,
            opening_exploration_rate=opening_exploration_rate,
            opening_exploration_radius=opening_exploration_radius,
        )

        if used_opening_exploration:
            exploration_count += 1

        policy_target = build_policy_target(board_size=board_size, move=move)

        samples.append(
            SelfPlaySample(
                state=state,
                policy_target=policy_target,
                current_player=current_player,
            )
        )

        game.play_move(move)

    winner = get_winner_from_status(game.status)

    # If max_moves is reached before a terminal result, treat it as draw.
    if game.status == GameStatus.ONGOING:
        winner = Board.EMPTY

    for sample in samples:
        if winner == Board.EMPTY:
            sample.value_target = 0.0
        elif sample.current_player == winner:
            sample.value_target = 1.0
        else:
            sample.value_target = -1.0

    return samples, winner, game.status, exploration_count


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

    print(f"Saved self-play dataset to: {output_path}")
    print(f"states shape: {states.shape}")
    print(f"policy_targets shape: {policy_targets.shape}")
    print(f"value_targets shape: {value_targets.shape}")
    print(f"current_players shape: {current_players.shape}")


def build_default_output_path(
    output_dir: Path,
    agent: str,
    rule: str,
    board_size: int,
    games: int,
    seed: int,
    opening_exploration_moves: int,
    opening_exploration_rate: float,
) -> Path:
    """Build a readable default output filename."""

    exploration_suffix = ""

    if opening_exploration_moves > 0 and opening_exploration_rate > 0.0:
        rate_text = str(opening_exploration_rate).replace(".", "p")
        exploration_suffix = (
            f"_openexp{opening_exploration_moves}_rate{rate_text}"
        )

    filename = (
        f"self_play_{agent}_{rule}_{board_size}x{board_size}"
        f"_{games}games_seed{seed}{exploration_suffix}.npz"
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

    # NN-MCTS options
    parser.add_argument("--nn-mcts-simulations", type=int, default=25)
    parser.add_argument("--nn-mcts-checkpoint", type=str, default=None)
    parser.add_argument("--nn-mcts-exploration-weight", type=float, default=1.5)
    parser.add_argument("--nn-mcts-candidate-radius", type=int, default=2)
    parser.add_argument("--nn-mcts-device", type=str, default="auto")

    # Opening exploration options
    parser.add_argument(
        "--opening-exploration-moves",
        type=int,
        default=0,
        help=(
            "Number of opening moves where local random exploration is allowed. "
            "For example, 8 means moves 2 to 8 may explore because move 1 is kept "
            "controlled by the agent. Use 0 to disable exploration."
        ),
    )

    parser.add_argument(
        "--opening-exploration-rate",
        type=float,
        default=0.0,
        help=(
            "Probability of using local random exploration during the opening stage. "
            "For example, 0.3 means 30 percent."
        ),
    )

    parser.add_argument(
        "--opening-exploration-radius",
        type=int,
        default=3,
        help=(
            "Radius around existing stones for local random opening exploration. "
            "This prevents full-board random moves."
        ),
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
        opening_exploration_moves=args.opening_exploration_moves,
        opening_exploration_rate=args.opening_exploration_rate,
    )

    all_samples: List[SelfPlaySample] = []
    game_lengths: List[int] = []
    winners: List[int] = []
    exploration_counts: List[int] = []

    print("Self-play configuration:")
    print(f"  games: {args.games}")
    print(f"  board_size: {args.board_size}")
    print(f"  rule: {args.rule}")
    print(f"  agent: {args.agent}")
    print(f"  max_moves: {args.max_moves}")
    print(f"  seed: {args.seed}")
    print(f"  nn_mcts_simulations: {args.nn_mcts_simulations}")
    print(f"  nn_mcts_checkpoint: {args.nn_mcts_checkpoint}")
    print(f"  nn_mcts_exploration_weight: {args.nn_mcts_exploration_weight}")
    print(f"  nn_mcts_candidate_radius: {args.nn_mcts_candidate_radius}")
    print(f"  nn_mcts_device: {args.nn_mcts_device}")
    print(f"  opening_exploration_moves: {args.opening_exploration_moves}")
    print(f"  opening_exploration_rate: {args.opening_exploration_rate}")
    print(f"  opening_exploration_radius: {args.opening_exploration_radius}")
    print()

    for game_index in tqdm(range(args.games), desc="Generating self-play games"):
        game_seed = args.seed + game_index

        samples, winner, status, exploration_count = play_self_play_game(
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
            opening_exploration_moves=args.opening_exploration_moves,
            opening_exploration_rate=args.opening_exploration_rate,
            opening_exploration_radius=args.opening_exploration_radius,
        )

        all_samples.extend(samples)
        game_lengths.append(len(samples))
        winners.append(winner)
        exploration_counts.append(exploration_count)

        print(
            f"Game {game_index + 1}/{args.games}: "
            f"{len(samples)} samples, "
            f"winner={winner}, "
            f"status={status.name}, "
            f"opening_explorations={exploration_count}"
        )

    metadata = {
        "games": args.games,
        "board_size": args.board_size,
        "rule": args.rule,
        "agent": args.agent,
        "seed": args.seed,
        "max_moves": args.max_moves,
        "nn_mcts_simulations": args.nn_mcts_simulations,
        "nn_mcts_checkpoint": args.nn_mcts_checkpoint,
        "nn_mcts_exploration_weight": args.nn_mcts_exploration_weight,
        "nn_mcts_candidate_radius": args.nn_mcts_candidate_radius,
        "nn_mcts_device": args.nn_mcts_device,
        "opening_exploration_moves": args.opening_exploration_moves,
        "opening_exploration_rate": args.opening_exploration_rate,
        "opening_exploration_radius": args.opening_exploration_radius,
        "game_lengths": game_lengths,
        "winners": winners,
        "exploration_counts": exploration_counts,
    }

    save_dataset(
        samples=all_samples,
        output_path=output_path,
        metadata=metadata,
    )

    print()
    print("Generation summary:")
    print(f"  total samples: {len(all_samples)}")
    print(f"  game lengths: {game_lengths}")
    print(f"  unique game lengths: {sorted(set(game_lengths))}")
    print(f"  exploration counts: {exploration_counts}")
    print(f"  total opening explorations: {sum(exploration_counts)}")
    print(f"  output: {output_path}")


if __name__ == "__main__":
    main()