from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

# Allow this script to be run directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from gomoku.agents import GreedyAgent, RandomAgent
from gomoku.board import Board
from gomoku.game import Game, GameStatus
from gomoku.neural_network import encode_board_state, move_to_action


@dataclass
class SelfPlaySample:
    """
    One training sample generated from self-play.

    state:
        Encoded board state before the move.
        Shape: [3, board_size, board_size]

    policy_target:
        One-hot vector showing the move actually played.
        Shape: [board_size * board_size]

    current_player:
        The player to move at this state.
        Board.BLACK or Board.WHITE.

    value_target:
        Filled after the game ends.
        +1 if current_player eventually wins.
        -1 if current_player eventually loses.
         0 if the game is a draw.
    """

    state: np.ndarray
    policy_target: np.ndarray
    current_player: int
    value_target: float = 0.0


def create_agent(agent_name: str, seed: int):
    """
    Create a simple agent for early self-play data generation.

    This is only for initial NN training pipeline validation.
    Later, NN-MCTS will generate stronger self-play data.
    """
    if agent_name == "random":
        return RandomAgent(seed=seed)

    if agent_name == "greedy":
        return GreedyAgent(seed=seed)

    raise ValueError(f"Unknown self-play agent: {agent_name}")


def create_policy_target(move: tuple[int, int], board_size: int) -> np.ndarray:
    """
    Create a one-hot policy target for the selected move.
    """
    target = np.zeros(board_size * board_size, dtype=np.float32)
    action = move_to_action(move, board_size)
    target[action] = 1.0
    return target


def get_value_for_player(winner: int | None, player: int) -> float:
    """
    Convert final winner into a value target from one player's perspective.
    """
    if winner is None:
        return 0.0

    if winner == player:
        return 1.0

    return -1.0


def play_one_self_play_game(
    board_size: int,
    rule_name: str,
    agent_name: str,
    seed: int,
    max_moves: int | None = None,
) -> list[SelfPlaySample]:
    """
    Run one self-play game and return training samples.

    The same type of agent is used for both sides.
    """
    game = Game(board_size=board_size, rule_name=rule_name)

    black_agent = create_agent(agent_name, seed=seed)
    white_agent = create_agent(agent_name, seed=seed + 1)

    samples: list[SelfPlaySample] = []

    move_limit = max_moves if max_moves is not None else board_size * board_size

    while not game.is_over() and len(game.board.move_history) < move_limit:
        current_player = game.current_player

        if current_player == Board.BLACK:
            agent = black_agent
        else:
            agent = white_agent

        state = encode_board_state(game.board, current_player)

        move = agent.select_move(game)
        policy_target = create_policy_target(move, board_size)

        samples.append(
            SelfPlaySample(
                state=state,
                policy_target=policy_target,
                current_player=current_player,
            )
        )

        game.play_move(move)

    winner = game.get_winner()

    for sample in samples:
        sample.value_target = get_value_for_player(winner, sample.current_player)

    return samples


def generate_self_play_dataset(
    num_games: int,
    board_size: int,
    rule_name: str,
    agent_name: str,
    seed: int,
    max_moves: int | None = None,
) -> list[SelfPlaySample]:
    """
    Generate self-play samples from multiple games.
    """
    all_samples: list[SelfPlaySample] = []

    random.seed(seed)
    np.random.seed(seed)

    for game_index in range(num_games):
        game_seed = seed + game_index * 2

        samples = play_one_self_play_game(
            board_size=board_size,
            rule_name=rule_name,
            agent_name=agent_name,
            seed=game_seed,
            max_moves=max_moves,
        )

        all_samples.extend(samples)

        print(
            f"Game {game_index + 1}/{num_games} completed: "
            f"{len(samples)} samples"
        )

    return all_samples


def save_dataset(
    samples: list[SelfPlaySample],
    output_path: Path,
    metadata: dict,
) -> None:
    """
    Save self-play samples as a compressed numpy dataset.
    """
    if not samples:
        raise ValueError("No samples to save.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    states = np.stack([sample.state for sample in samples]).astype(np.float32)
    policy_targets = np.stack(
        [sample.policy_target for sample in samples]
    ).astype(np.float32)
    value_targets = np.array(
        [[sample.value_target] for sample in samples],
        dtype=np.float32,
    )

    np.savez_compressed(
        output_path,
        states=states,
        policy_targets=policy_targets,
        value_targets=value_targets,
        metadata=json.dumps(metadata),
    )

    print(f"Saved dataset to: {output_path}")
    print(f"states shape: {states.shape}")
    print(f"policy_targets shape: {policy_targets.shape}")
    print(f"value_targets shape: {value_targets.shape}")


def build_default_output_path(
    output_dir: Path,
    agent_name: str,
    rule_name: str,
    board_size: int,
    num_games: int,
    seed: int,
) -> Path:
    """
    Build a readable default output path.
    """
    filename = (
        f"self_play_{agent_name}_{rule_name}_"
        f"{board_size}x{board_size}_{num_games}games_seed{seed}.npz"
    )

    return output_dir / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate self-play data for Gomoku NN training."
    )

    parser.add_argument(
        "--games",
        type=int,
        default=5,
        help="Number of self-play games to generate.",
    )

    parser.add_argument(
        "--board-size",
        type=int,
        default=15,
        help="Board size.",
    )

    parser.add_argument(
        "--rule",
        type=str,
        default="standard",
        choices=["standard", "pro"],
        help="Rule used for self-play data generation.",
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="greedy",
        choices=["random", "greedy"],
        help="Agent used to generate initial self-play data.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed.",
    )

    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="Optional maximum number of moves per game.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/self_play"),
        help="Directory to save generated self-play dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output file path.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.games <= 0:
        raise ValueError("--games must be positive.")

    if args.board_size <= 0:
        raise ValueError("--board-size must be positive.")

    output_path = args.output

    if output_path is None:
        output_path = build_default_output_path(
            output_dir=args.output_dir,
            agent_name=args.agent,
            rule_name=args.rule,
            board_size=args.board_size,
            num_games=args.games,
            seed=args.seed,
        )

    samples = generate_self_play_dataset(
        num_games=args.games,
        board_size=args.board_size,
        rule_name=args.rule,
        agent_name=args.agent,
        seed=args.seed,
        max_moves=args.max_moves,
    )

    metadata = {
        "num_games": args.games,
        "board_size": args.board_size,
        "rule": args.rule,
        "agent": args.agent,
        "seed": args.seed,
        "max_moves": args.max_moves,
        "num_samples": len(samples),
    }

    save_dataset(
        samples=samples,
        output_path=output_path,
        metadata=metadata,
    )