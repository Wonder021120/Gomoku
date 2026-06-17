from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gomoku.board import Board, Move
from gomoku.minimax_agent import MinimaxAgent

if TYPE_CHECKING:
    from gomoku.agents import BaseAgent
    from gomoku.game import Game


@dataclass
class Swap2OpeningResult:
    """
    Stores the result of the automated Swap2 opening protocol.
    """

    choice: str
    opening_template: str
    opening_moves: list[tuple[int, Move]]
    final_black_agent_name: str
    final_white_agent_name: str

    @property
    def opening_moves_text(self) -> str:
        parts = []

        for colour, move in self.opening_moves:
            colour_name = "B" if colour == Board.BLACK else "W"
            row, col = move
            parts.append(f"{colour_name}({row},{col})")

        return "; ".join(parts)


def run_swap2_opening(
    game: "Game",
    slicer_agent: "BaseAgent",
    chooser_agent: "BaseAgent",
    choice_threshold: float = 1000.0,
) -> tuple[Swap2OpeningResult, "BaseAgent", "BaseAgent"]:
    """
    Run an automated AI-vs-AI Swap2 opening protocol.

    Swap2 rule concept:
    - The slicer proposes an opening with 2 black stones and 1 white stone.
    - The chooser may choose black, choose white, or ask to add two more stones.
    - If two more stones are added, the slicer chooses the final colour.
    - The game then continues normally.

    In this project, the opening stones are generated using a fixed central
    template. This is not a rule restriction. It is an experimental design choice
    used to make AI-vs-AI experiments reproducible and comparable.
    """
    if len(game.board.move_history) != 0:
        raise ValueError("Swap2 opening must start from an empty board.")

    opening_moves: list[tuple[int, Move]] = []

    # First proposal: 2 black + 1 white.
    #
    # For 15x15:
    # B: (7, 7) centre
    # W: (7, 8) right of centre
    # B: (8, 7) below centre
    initial_template = _get_initial_three_stone_template(game.board.size)

    for colour, move in initial_template:
        _place_forced_stone(game, move, colour)
        opening_moves.append((colour, move))

    initial_black_advantage = _evaluate_black_advantage(game)

    # If black is clearly better, chooser takes black.
    if initial_black_advantage > choice_threshold:
        result = Swap2OpeningResult(
            choice="chooser_choose_black",
            opening_template="central_3_stone_template",
            opening_moves=opening_moves,
            final_black_agent_name=chooser_agent.name,
            final_white_agent_name=slicer_agent.name,
        )
        game.current_player = Board.WHITE
        return result, chooser_agent, slicer_agent

    # If white is clearly better, chooser takes white.
    if initial_black_advantage < -choice_threshold:
        result = Swap2OpeningResult(
            choice="chooser_choose_white",
            opening_template="central_3_stone_template",
            opening_moves=opening_moves,
            final_black_agent_name=slicer_agent.name,
            final_white_agent_name=chooser_agent.name,
        )
        game.current_player = Board.WHITE
        return result, slicer_agent, chooser_agent

    # If the position is relatively balanced, chooser adds two more stones:
    # one white and one black.
    #
    # For 15x15:
    # W: (6, 7) above centre
    # B: (7, 6) left of centre
    additional_template = _get_additional_two_stone_template(game.board.size)

    for colour, move in additional_template:
        _place_forced_stone(game, move, colour)
        opening_moves.append((colour, move))

    final_black_advantage = _evaluate_black_advantage(game)

    # After five stones, the slicer chooses the final colour.
    if final_black_advantage >= 0:
        result = Swap2OpeningResult(
            choice="add_two_then_slicer_choose_black",
            opening_template="central_5_stone_template",
            opening_moves=opening_moves,
            final_black_agent_name=slicer_agent.name,
            final_white_agent_name=chooser_agent.name,
        )
        game.current_player = Board.WHITE
        return result, slicer_agent, chooser_agent

    result = Swap2OpeningResult(
        choice="add_two_then_slicer_choose_white",
        opening_template="central_5_stone_template",
        opening_moves=opening_moves,
        final_black_agent_name=chooser_agent.name,
        final_white_agent_name=slicer_agent.name,
    )
    game.current_player = Board.WHITE
    return result, chooser_agent, slicer_agent


def _get_initial_three_stone_template(board_size: int) -> list[tuple[int, Move]]:
    """
    Return the fixed central 3-stone Swap2 proposal.

    This is one legal Swap2 proposal used for controlled experiments.
    """
    centre = board_size // 2

    return [
        (Board.BLACK, (centre, centre)),
        (Board.WHITE, (centre, centre + 1)),
        (Board.BLACK, (centre + 1, centre)),
    ]


def _get_additional_two_stone_template(board_size: int) -> list[tuple[int, Move]]:
    """
    Return the fixed additional 2-stone Swap2 proposal.
    """
    centre = board_size // 2

    return [
        (Board.WHITE, (centre - 1, centre)),
        (Board.BLACK, (centre, centre - 1)),
    ]


def _place_forced_stone(game: "Game", move: Move, colour: int) -> None:
    """
    Place a forced opening stone without alternating turns.

    Swap2 opening placement is not normal alternating play, so this directly
    modifies the board while still using Board.place_stone validation.
    """
    row, col = move
    game.board.place_stone(row, col, colour)
    game.last_move = move


def _evaluate_black_advantage(game: "Game") -> float:
    """
    Estimate whether the current position favours black or white.

    Positive score: black is better.
    Negative score: white is better.
    """
    evaluator = MinimaxAgent(depth=1, candidate_radius=1)

    return float(evaluator._evaluate_board(game.board, Board.BLACK))