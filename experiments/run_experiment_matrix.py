from __future__ import annotations

import sys
from pathlib import Path

# Allow this script to be run directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from experiments.run_tournament import build_default_output_path, run_tournament


def run_validation_matrix() -> None:
    """
    Run a small validation experiment matrix.

    This is not the final formal experiment.
    It is used to check that all rules and major agents can run through the
    same tournament pipeline.
    """
    rules = ["standard", "pro", "swap2"]

    same_agent_matches = [
        ("random", "random"),
        ("greedy", "greedy"),
        ("minimax", "minimax"),
        ("mcts", "mcts"),
    ]

    strength_matches = [
        ("greedy", "random"),
        ("random", "greedy"),
        ("minimax", "greedy"),
        ("greedy", "minimax"),
        ("mcts", "minimax"),
        ("minimax", "mcts"),
    ]

    games = 3
    board_size = 15
    seed = 2026

    minimax_depth = 1
    minimax_candidate_radius = 1

    mcts_simulations = 10
    mcts_exploration_weight = 1.4
    mcts_candidate_radius = 1
    mcts_rollout_depth = 10

    print("Running same-agent validation experiments...")

    for rule in rules:
        for black_agent, white_agent in same_agent_matches:
            output_path = build_default_output_path(
                black_agent=black_agent,
                white_agent=white_agent,
                rule_name=rule,
                board_size=board_size,
                minimax_depth=minimax_depth,
                minimax_candidate_radius=minimax_candidate_radius,
                mcts_simulations=mcts_simulations,
                mcts_candidate_radius=mcts_candidate_radius,
                mcts_rollout_depth=mcts_rollout_depth,
            )

            run_tournament(
                games=games,
                board_size=board_size,
                rule_name=rule,
                black_agent_name=black_agent,
                white_agent_name=white_agent,
                seed=seed,
                output_path=output_path,
                minimax_depth=minimax_depth,
                minimax_candidate_radius=minimax_candidate_radius,
                mcts_simulations=mcts_simulations,
                mcts_exploration_weight=mcts_exploration_weight,
                mcts_candidate_radius=mcts_candidate_radius,
                mcts_rollout_depth=mcts_rollout_depth,
            )

    print("Running AI-strength validation experiments...")

    for rule in rules:
        for black_agent, white_agent in strength_matches:
            output_path = build_default_output_path(
                black_agent=black_agent,
                white_agent=white_agent,
                rule_name=rule,
                board_size=board_size,
                minimax_depth=minimax_depth,
                minimax_candidate_radius=minimax_candidate_radius,
                mcts_simulations=mcts_simulations,
                mcts_candidate_radius=mcts_candidate_radius,
                mcts_rollout_depth=mcts_rollout_depth,
            )

            run_tournament(
                games=games,
                board_size=board_size,
                rule_name=rule,
                black_agent_name=black_agent,
                white_agent_name=white_agent,
                seed=seed,
                output_path=output_path,
                minimax_depth=minimax_depth,
                minimax_candidate_radius=minimax_candidate_radius,
                mcts_simulations=mcts_simulations,
                mcts_exploration_weight=mcts_exploration_weight,
                mcts_candidate_radius=mcts_candidate_radius,
                mcts_rollout_depth=mcts_rollout_depth,
            )

    print("Validation experiment matrix finished.")


if __name__ == "__main__":
    run_validation_matrix()