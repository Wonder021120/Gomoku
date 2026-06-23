from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

# Allow this script to be run directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from gomoku.agents import GreedyAgent, RandomAgent
from gomoku.mcts_agent import MCTSAgent
from gomoku.minimax_agent import MinimaxAgent
from gomoku.nn_mcts_agent import NNMCTSAgent

from experiments.run_match import MatchResult, run_match


def create_agent(
    agent_name: str,
    seed: int,
    board_size: int,
    minimax_depth: int,
    minimax_candidate_radius: int,
    mcts_simulations: int,
    mcts_exploration_weight: float,
    mcts_candidate_radius: int,
    mcts_rollout_depth: int,
    nn_mcts_simulations: int,
    nn_mcts_checkpoint: str | None,
    nn_mcts_exploration_weight: float,
    nn_mcts_candidate_radius: int,
    nn_mcts_device: str,
):
    """
    Create an AI agent by name.
    """
    if agent_name == "random":
        return RandomAgent(seed=seed)

    if agent_name == "greedy":
        return GreedyAgent(seed=seed)

    if agent_name == "minimax":
        return MinimaxAgent(
            depth=minimax_depth,
            candidate_radius=minimax_candidate_radius,
            seed=seed,
        )

    if agent_name == "mcts":
        return MCTSAgent(
            simulations=mcts_simulations,
            exploration_weight=mcts_exploration_weight,
            candidate_radius=mcts_candidate_radius,
            rollout_depth_limit=mcts_rollout_depth,
            seed=seed,
        )

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

    raise ValueError(f"Unknown agent: {agent_name}")


def build_agent_config(
    agent_name: str,
    minimax_depth: int,
    minimax_candidate_radius: int,
    mcts_simulations: int,
    mcts_exploration_weight: float,
    mcts_candidate_radius: int,
    mcts_rollout_depth: int,
    nn_mcts_simulations: int,
    nn_mcts_checkpoint: str | None,
    nn_mcts_exploration_weight: float,
    nn_mcts_candidate_radius: int,
    nn_mcts_device: str,
) -> str:
    """
    Build a readable config string for CSV logging.
    """
    if agent_name == "random":
        return "random"

    if agent_name == "greedy":
        return "greedy"

    if agent_name == "minimax":
        return (
            f"depth={minimax_depth};"
            f"candidate_radius={minimax_candidate_radius}"
        )

    if agent_name == "mcts":
        return (
            f"simulations={mcts_simulations};"
            f"exploration_weight={mcts_exploration_weight};"
            f"candidate_radius={mcts_candidate_radius};"
            f"rollout_depth={mcts_rollout_depth}"
        )

    if agent_name == "nn_mcts":
        checkpoint_text = nn_mcts_checkpoint if nn_mcts_checkpoint is not None else "none"

        return (
            f"simulations={nn_mcts_simulations};"
            f"checkpoint={checkpoint_text};"
            f"exploration_weight={nn_mcts_exploration_weight};"
            f"candidate_radius={nn_mcts_candidate_radius};"
            f"device={nn_mcts_device}"
        )

    return "unknown"


def run_tournament(
    games: int,
    board_size: int,
    rule_name: str,
    black_agent_name: str,
    white_agent_name: str,
    seed: int,
    minimax_depth: int,
    minimax_candidate_radius: int,
    mcts_simulations: int,
    mcts_exploration_weight: float,
    mcts_candidate_radius: int,
    mcts_rollout_depth: int,
    nn_mcts_simulations: int,
    nn_mcts_checkpoint: str | None,
    nn_mcts_exploration_weight: float,
    nn_mcts_candidate_radius: int,
    nn_mcts_device: str,
) -> list[MatchResult]:
    """
    Run multiple games and return match results.
    """
    if games <= 0:
        raise ValueError("games must be positive.")

    results: list[MatchResult] = []

    for game_index in range(games):
        game_seed = seed + game_index * 100

        black_agent = create_agent(
            agent_name=black_agent_name,
            seed=game_seed,
            board_size=board_size,
            minimax_depth=minimax_depth,
            minimax_candidate_radius=minimax_candidate_radius,
            mcts_simulations=mcts_simulations,
            mcts_exploration_weight=mcts_exploration_weight,
            mcts_candidate_radius=mcts_candidate_radius,
            mcts_rollout_depth=mcts_rollout_depth,
            nn_mcts_simulations=nn_mcts_simulations,
            nn_mcts_checkpoint=nn_mcts_checkpoint,
            nn_mcts_exploration_weight=nn_mcts_exploration_weight,
            nn_mcts_candidate_radius=nn_mcts_candidate_radius,
            nn_mcts_device=nn_mcts_device,
        )

        white_agent = create_agent(
            agent_name=white_agent_name,
            seed=game_seed + 1,
            board_size=board_size,
            minimax_depth=minimax_depth,
            minimax_candidate_radius=minimax_candidate_radius,
            mcts_simulations=mcts_simulations,
            mcts_exploration_weight=mcts_exploration_weight,
            mcts_candidate_radius=mcts_candidate_radius,
            mcts_rollout_depth=mcts_rollout_depth,
            nn_mcts_simulations=nn_mcts_simulations,
            nn_mcts_checkpoint=nn_mcts_checkpoint,
            nn_mcts_exploration_weight=nn_mcts_exploration_weight,
            nn_mcts_candidate_radius=nn_mcts_candidate_radius,
            nn_mcts_device=nn_mcts_device,
        )

        result = run_match(
            black_agent=black_agent,
            white_agent=white_agent,
            board_size=board_size,
            rule_name=rule_name,
            verbose=False,
        )

        results.append(result)

        print(
            f"Game {game_index + 1}/{games} completed | "
            f"winner={result.winner} | "
            f"moves={result.moves}"
        )

    return results


def save_results_to_csv(
    results: list[MatchResult],
    output_path: Path,
    rule_name: str,
    black_agent_config: str,
    white_agent_config: str,
) -> None:
    """
    Save tournament results to CSV.
    """
    if not results:
        raise ValueError("No results to save.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "game_id",
        "rule",
        "board_size",
        "black_agent",
        "white_agent",
        "black_agent_config",
        "white_agent_config",
        "swap2_choice",
        "swap2_opening_template",
        "swap2_opening_moves",
        "status",
        "winner",
        "first_player_win",
        "moves",
        "black_total_time",
        "white_total_time",
        "black_avg_time",
        "white_avg_time",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for game_id, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "game_id": game_id,
                    "rule": rule_name,
                    "board_size": result.board_size,
                    "black_agent": result.black_agent,
                    "white_agent": result.white_agent,
                    "black_agent_config": black_agent_config,
                    "white_agent_config": white_agent_config,
                    "swap2_choice": result.swap2_choice,
                    "swap2_opening_template": result.swap2_opening_template,
                    "swap2_opening_moves": result.swap2_opening_moves,
                    "status": result.status,
                    "winner": result.winner,
                    "first_player_win": result.first_player_win,
                    "moves": result.moves,
                    "black_total_time": result.black_total_time,
                    "white_total_time": result.white_total_time,
                    "black_avg_time": result.black_avg_time,
                    "white_avg_time": result.white_avg_time,
                }
            )

    print(f"Saved results to: {output_path}")


def build_default_output_path(
    output_dir: Path,
    rule_name: str,
    black_agent_name: str,
    white_agent_name: str,
    games: int,
    board_size: int,
    seed: int,
) -> Path:
    """
    Build default output path for tournament CSV.
    """
    filename = (
        f"{rule_name}_{black_agent_name}_vs_{white_agent_name}_"
        f"{board_size}x{board_size}_{games}games_seed{seed}.csv"
    )

    return output_dir / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gomoku tournaments between AI agents."
    )

    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games to run.",
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
        choices=["standard", "pro", "swap2"],
        help="Opening rule.",
    )

    parser.add_argument(
        "--black",
        type=str,
        default="random",
        choices=["random", "greedy", "minimax", "mcts", "nn_mcts"],
        help="Black agent.",
    )

    parser.add_argument(
        "--white",
        type=str,
        default="random",
        choices=["random", "greedy", "minimax", "mcts", "nn_mcts"],
        help="White agent.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/raw"),
        help="Directory to save raw CSV results.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output CSV path.",
    )

    parser.add_argument(
        "--minimax-depth",
        type=int,
        default=2,
        help="Minimax search depth.",
    )

    parser.add_argument(
        "--minimax-candidate-radius",
        type=int,
        default=1,
        help="Minimax candidate radius.",
    )

    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=50,
        help="MCTS simulations per move.",
    )

    parser.add_argument(
        "--mcts-exploration-weight",
        type=float,
        default=1.4,
        help="MCTS exploration weight.",
    )

    parser.add_argument(
        "--mcts-candidate-radius",
        type=int,
        default=1,
        help="MCTS candidate radius.",
    )

    parser.add_argument(
        "--mcts-rollout-depth",
        type=int,
        default=30,
        help="MCTS rollout depth limit.",
    )

    parser.add_argument(
        "--nn-mcts-simulations",
        type=int,
        default=25,
        help="NN-MCTS simulations per move.",
    )

    parser.add_argument(
        "--nn-mcts-checkpoint",
        type=str,
        default=None,
        help="Optional NN-MCTS checkpoint path.",
    )

    parser.add_argument(
        "--nn-mcts-exploration-weight",
        type=float,
        default=1.5,
        help="NN-MCTS PUCT exploration weight.",
    )

    parser.add_argument(
        "--nn-mcts-candidate-radius",
        type=int,
        default=2,
        help="NN-MCTS candidate radius.",
    )

    parser.add_argument(
        "--nn-mcts-device",
        type=str,
        default="auto",
        help="NN-MCTS device: auto, cpu, or cuda.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    black_agent_config = build_agent_config(
        agent_name=args.black,
        minimax_depth=args.minimax_depth,
        minimax_candidate_radius=args.minimax_candidate_radius,
        mcts_simulations=args.mcts_simulations,
        mcts_exploration_weight=args.mcts_exploration_weight,
        mcts_candidate_radius=args.mcts_candidate_radius,
        mcts_rollout_depth=args.mcts_rollout_depth,
        nn_mcts_simulations=args.nn_mcts_simulations,
        nn_mcts_checkpoint=args.nn_mcts_checkpoint,
        nn_mcts_exploration_weight=args.nn_mcts_exploration_weight,
        nn_mcts_candidate_radius=args.nn_mcts_candidate_radius,
        nn_mcts_device=args.nn_mcts_device,
    )

    white_agent_config = build_agent_config(
        agent_name=args.white,
        minimax_depth=args.minimax_depth,
        minimax_candidate_radius=args.minimax_candidate_radius,
        mcts_simulations=args.mcts_simulations,
        mcts_exploration_weight=args.mcts_exploration_weight,
        mcts_candidate_radius=args.mcts_candidate_radius,
        mcts_rollout_depth=args.mcts_rollout_depth,
        nn_mcts_simulations=args.nn_mcts_simulations,
        nn_mcts_checkpoint=args.nn_mcts_checkpoint,
        nn_mcts_exploration_weight=args.nn_mcts_exploration_weight,
        nn_mcts_candidate_radius=args.nn_mcts_candidate_radius,
        nn_mcts_device=args.nn_mcts_device,
    )

    output_path = args.output

    if output_path is None:
        output_path = build_default_output_path(
            output_dir=args.output_dir,
            rule_name=args.rule,
            black_agent_name=args.black,
            white_agent_name=args.white,
            games=args.games,
            board_size=args.board_size,
            seed=args.seed,
        )

    tournament_results = run_tournament(
        games=args.games,
        board_size=args.board_size,
        rule_name=args.rule,
        black_agent_name=args.black,
        white_agent_name=args.white,
        seed=args.seed,
        minimax_depth=args.minimax_depth,
        minimax_candidate_radius=args.minimax_candidate_radius,
        mcts_simulations=args.mcts_simulations,
        mcts_exploration_weight=args.mcts_exploration_weight,
        mcts_candidate_radius=args.mcts_candidate_radius,
        mcts_rollout_depth=args.mcts_rollout_depth,
        nn_mcts_simulations=args.nn_mcts_simulations,
        nn_mcts_checkpoint=args.nn_mcts_checkpoint,
        nn_mcts_exploration_weight=args.nn_mcts_exploration_weight,
        nn_mcts_candidate_radius=args.nn_mcts_candidate_radius,
        nn_mcts_device=args.nn_mcts_device,
    )

    save_results_to_csv(
        results=tournament_results,
        output_path=output_path,
        rule_name=args.rule,
        black_agent_config=black_agent_config,
        white_agent_config=white_agent_config,
    )