from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from gomoku.agents import BaseAgent
from gomoku.board import Board, Move
from gomoku.game import Game, GameStatus


WIN_SCORE = 1_000_000


@dataclass
class MinimaxAgent(BaseAgent):
    """
    A Minimax agent with Alpha-Beta pruning.

    This is the first classical AI baseline for the project.
    The evaluation function uses simple Gomoku pattern scoring:
    five, open four, blocked four, open three, blocked three, etc.
    """

    depth: int = 2
    candidate_radius: int = 2
    seed: Optional[int] = None
    name: str = "minimax"

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("Search depth must be positive.")
        if self.candidate_radius <= 0:
            raise ValueError("Candidate radius must be positive.")
        self.rng = random.Random(self.seed)

    def select_move(self, game: Game) -> Move:
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            raise ValueError("No legal moves available.")

        # On an empty board, play the centre. This reduces the opening branching factor.
        if len(game.board.move_history) == 0:
            centre = game.board.size // 2
            return (centre, centre)

        root_player = game.current_player
        best_score = -math.inf
        best_moves: list[Move] = []

        candidate_moves = self._get_candidate_moves(game)

        for move in candidate_moves:
            simulated_game = game.copy()
            simulated_game.play_move(move)

            score = self._minimax(
                game=simulated_game,
                depth=self.depth - 1,
                alpha=-math.inf,
                beta=math.inf,
                root_player=root_player,
            )

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return self.rng.choice(best_moves)

    def _minimax(
        self,
        game: Game,
        depth: int,
        alpha: float,
        beta: float,
        root_player: int,
    ) -> float:
        """
        Minimax search with alpha-beta pruning.
        """
        if game.is_over() or depth == 0:
            return self._evaluate_game(game, root_player)

        candidate_moves = self._get_candidate_moves(game)

        if not candidate_moves:
            return self._evaluate_game(game, root_player)

        maximizing = game.current_player == root_player

        if maximizing:
            value = -math.inf

            for move in candidate_moves:
                simulated_game = game.copy()
                simulated_game.play_move(move)

                value = max(
                    value,
                    self._minimax(
                        game=simulated_game,
                        depth=depth - 1,
                        alpha=alpha,
                        beta=beta,
                        root_player=root_player,
                    ),
                )

                alpha = max(alpha, value)

                if alpha >= beta:
                    break

            return value

        value = math.inf

        for move in candidate_moves:
            simulated_game = game.copy()
            simulated_game.play_move(move)

            value = min(
                value,
                self._minimax(
                    game=simulated_game,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    root_player=root_player,
                ),
            )

            beta = min(beta, value)

            if alpha >= beta:
                break

        return value

    def _evaluate_game(self, game: Game, player: int) -> float:
        """
        Evaluate the current game state from the perspective of player.
        """
        winner = game.get_winner()

        if winner == player:
            return WIN_SCORE

        if winner == self._opponent(player):
            return -WIN_SCORE

        if game.status == GameStatus.DRAW:
            return 0.0

        return self._evaluate_board(game.board, player)

    def _evaluate_board(self, board: Board, player: int) -> float:
        """
        Evaluate the board using Gomoku pattern scores.

        Positive score means the position is good for player.
        Negative score means the position is good for the opponent.
        """
        opponent = self._opponent(player)

        player_score = self._score_player_patterns(board, player)
        opponent_score = self._score_player_patterns(board, opponent)

        # Opponent threats are weighted slightly higher so the agent is more defensive.
        return player_score - 1.1 * opponent_score

    def _score_player_patterns(self, board: Board, player: int) -> float:
        """
        Score all consecutive stone patterns for one player.
        """
        total_score = 0.0

        for line in self._collect_lines(board):
            total_score += self._score_line(line, player)

        return total_score

    def _score_line(self, line: list[int], player: int) -> float:
        """
        Score consecutive runs of player's stones in a single line.

        A run is evaluated by:
        - length of the consecutive stones
        - number of open ends

        open ends:
        - 2 means both ends are empty
        - 1 means one end is empty
        - 0 means both ends are blocked or outside the board
        """
        score = 0.0
        index = 0
        line_length = len(line)

        while index < line_length:
            if line[index] != player:
                index += 1
                continue

            start = index

            while index < line_length and line[index] == player:
                index += 1

            end = index - 1
            run_length = end - start + 1

            left_open = start - 1 >= 0 and line[start - 1] == Board.EMPTY
            right_open = end + 1 < line_length and line[end + 1] == Board.EMPTY
            open_ends = int(left_open) + int(right_open)

            score += self._score_run(run_length, open_ends)

        return score

    def _score_run(self, run_length: int, open_ends: int) -> float:
        """
        Score a consecutive run according to simple Gomoku patterns.
        """
        if run_length >= 5:
            return WIN_SCORE

        if run_length == 4:
            if open_ends == 2:
                return 100_000  # open four
            if open_ends == 1:
                return 10_000  # blocked four
            return 0

        if run_length == 3:
            if open_ends == 2:
                return 5_000  # open three
            if open_ends == 1:
                return 1_000  # blocked three
            return 0

        if run_length == 2:
            if open_ends == 2:
                return 300  # open two
            if open_ends == 1:
                return 50  # blocked two
            return 0

        if run_length == 1:
            if open_ends == 2:
                return 10
            if open_ends == 1:
                return 1
            return 0

        return 0

    def _collect_lines(self, board: Board) -> list[list[int]]:
        """
        Collect all rows, columns, and diagonals.
        """
        lines: list[list[int]] = []

        size = board.size

        # Rows
        for row in range(size):
            lines.append([int(board.grid[row, col]) for col in range(size)])

        # Columns
        for col in range(size):
            lines.append([int(board.grid[row, col]) for row in range(size)])

        # Diagonals: top-left to bottom-right
        for start_col in range(size):
            line = []
            row, col = 0, start_col
            while row < size and col < size:
                line.append(int(board.grid[row, col]))
                row += 1
                col += 1
            lines.append(line)

        for start_row in range(1, size):
            line = []
            row, col = start_row, 0
            while row < size and col < size:
                line.append(int(board.grid[row, col]))
                row += 1
                col += 1
            lines.append(line)

        # Diagonals: top-right to bottom-left
        for start_col in range(size):
            line = []
            row, col = 0, start_col
            while row < size and col >= 0:
                line.append(int(board.grid[row, col]))
                row += 1
                col -= 1
            lines.append(line)

        for start_row in range(1, size):
            line = []
            row, col = start_row, size - 1
            while row < size and col >= 0:
                line.append(int(board.grid[row, col]))
                row += 1
                col -= 1
            lines.append(line)

        return lines

    def _get_candidate_moves(self, game: Game) -> list[Move]:
        """
        Generate candidate moves near existing stones.

        Searching every empty cell on a 15x15 board is expensive.
        This method only considers empty cells within candidate_radius
        of an existing stone.
        """
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return []

        if len(game.board.move_history) == 0:
            centre = game.board.size // 2
            return [(centre, centre)]

        candidates: set[Move] = set()
        size = game.board.size

        for stone_position, _player in game.board.move_history:
            row, col = stone_position

            for delta_row in range(-self.candidate_radius, self.candidate_radius + 1):
                for delta_col in range(-self.candidate_radius, self.candidate_radius + 1):
                    candidate_row = row + delta_row
                    candidate_col = col + delta_col

                    if (
                        0 <= candidate_row < size
                        and 0 <= candidate_col < size
                        and game.board.is_empty(candidate_row, candidate_col)
                    ):
                        candidates.add((candidate_row, candidate_col))

        if not candidates:
            return legal_moves

        centre = size // 2

        return sorted(
            candidates,
            key=lambda move: abs(move[0] - centre) + abs(move[1] - centre),
        )

    def _opponent(self, player: int) -> int:
        return Board.WHITE if player == Board.BLACK else Board.BLACK