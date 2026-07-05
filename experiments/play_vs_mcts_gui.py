from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gomoku.board import Board
from gomoku.game import Game, GameStatus
from gomoku.mcts_agent import MCTSAgent


class GomokuMCTSGUI:
    def __init__(
        self,
        board_size: int = 9,
        cell_size: int = 56,
        mcts_simulations: int = 200,
    ) -> None:
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = 36
        self.canvas_size = self.margin * 2 + self.cell_size * (self.board_size - 1)

        self.root = tk.Tk()
        self.root.title("Gomoku: Human vs MCTS")

        self.game = Game(board_size=self.board_size, rule_name="standard")
        self.mcts_agent = MCTSAgent(simulations=mcts_simulations, seed=2026)

        self.human_player = Board.BLACK
        self.ai_player = Board.WHITE
        self.game_over = False
        self.ai_thinking = False

        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=8)

        self.status_label = tk.Label(
            top_frame,
            text="Human: Black | MCTS: White",
            font=("Arial", 13),
        )
        self.status_label.pack()

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=4)

        self.black_button = tk.Button(
            button_frame,
            text="New game: Human Black",
            command=self.new_game_human_black,
            width=20,
        )
        self.black_button.grid(row=0, column=0, padx=5)

        self.white_button = tk.Button(
            button_frame,
            text="New game: Human White",
            command=self.new_game_human_white,
            width=20,
        )
        self.white_button.grid(row=0, column=1, padx=5)

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#DDBB77",
        )
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)

        self.draw_board()
        self.update_status()

    def run(self) -> None:
        self.root.mainloop()

    def new_game_human_black(self) -> None:
        self.human_player = Board.BLACK
        self.ai_player = Board.WHITE
        self.reset_game()
        self.update_status("New game. You are Black. Your move.")

    def new_game_human_white(self) -> None:
        self.human_player = Board.WHITE
        self.ai_player = Board.BLACK
        self.reset_game()
        self.update_status("New game. You are White. MCTS is thinking...")
        self.root.after(300, self.ai_move)

    def reset_game(self) -> None:
        self.game = Game(board_size=self.board_size, rule_name="standard")
        self.game_over = False
        self.ai_thinking = False
        self.draw_board()

    def draw_board(self) -> None:
        self.canvas.delete("all")

        for index in range(self.board_size):
            start = self.margin
            end = self.margin + self.cell_size * (self.board_size - 1)
            pos = self.margin + index * self.cell_size

            self.canvas.create_line(start, pos, end, pos, width=2)
            self.canvas.create_line(pos, start, pos, end, width=2)

        centre = self.board_size // 2
        self.draw_star_point(centre, centre)

        for row in range(self.board_size):
            for col in range(self.board_size):
                stone = int(self.game.board.grid[row, col])

                if stone == Board.BLACK:
                    self.draw_stone(row, col, "black")
                elif stone == Board.WHITE:
                    self.draw_stone(row, col, "white")

    def draw_star_point(self, row: int, col: int) -> None:
        x, y = self.board_to_pixel(row, col)
        radius = 4
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="black",
            outline="black",
        )

    def draw_stone(self, row: int, col: int, color: str) -> None:
        x, y = self.board_to_pixel(row, col)
        radius = self.cell_size * 0.38

        outline = "black"
        if color == "white":
            outline = "#333333"

        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=color,
            outline=outline,
            width=2,
        )

    def board_to_pixel(self, row: int, col: int) -> tuple[int, int]:
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        return x, y

    def pixel_to_board(self, x: int, y: int) -> tuple[int, int] | None:
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)

        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return None

        px, py = self.board_to_pixel(row, col)
        distance_squared = (x - px) ** 2 + (y - py) ** 2

        if distance_squared > (self.cell_size * 0.42) ** 2:
            return None

        return row, col

    def on_click(self, event: tk.Event) -> None:
        if self.game_over or self.ai_thinking:
            return

        if self.game.current_player != self.human_player:
            return

        move = self.pixel_to_board(event.x, event.y)

        if move is None:
            return

        if move not in self.game.get_legal_moves():
            return

        self.play_human_move(move)

    def play_human_move(self, move: tuple[int, int]) -> None:
        try:
            self.game.play_move(move)
        except ValueError as exc:
            messagebox.showwarning("Illegal move", str(exc))
            return

        self.draw_board()

        if self.check_game_over():
            return

        self.update_status("MCTS is thinking...")
        self.ai_thinking = True
        self.root.after(200, self.ai_move)

    def ai_move(self) -> None:
        if self.game_over:
            return

        if self.game.current_player != self.ai_player:
            return

        try:
            move = self.mcts_agent.select_move(self.game)
            self.game.play_move(move)
        except Exception as exc:
            self.ai_thinking = False
            messagebox.showerror("MCTS error", str(exc))
            return

        self.ai_thinking = False
        self.draw_board()

        if self.check_game_over():
            return

        self.update_status("Your move.")

    def check_game_over(self) -> bool:
        if self.game.status == GameStatus.ONGOING:
            return False

        self.game_over = True

        if self.game.status == GameStatus.BLACK_WIN:
            winner_text = "Black wins"
        elif self.game.status == GameStatus.WHITE_WIN:
            winner_text = "White wins"
        else:
            winner_text = "Draw"

        if self.game.get_winner() == self.human_player:
            result_text = f"{winner_text}. You win!"
        elif self.game.get_winner() == self.ai_player:
            result_text = f"{winner_text}. MCTS wins."
        else:
            result_text = "Draw."

        self.update_status(result_text)
        messagebox.showinfo("Game over", result_text)
        return True

    def update_status(self, text: str | None = None) -> None:
        if text is None:
            human_color = "Black" if self.human_player == Board.BLACK else "White"
            ai_color = "White" if self.ai_player == Board.WHITE else "Black"
            text = f"Human: {human_color} | MCTS: {ai_color}"

        self.status_label.config(text=text)


def main() -> None:
    app = GomokuMCTSGUI(
        board_size=9,
        cell_size=56,
        mcts_simulations=200,
    )
    app.run()


if __name__ == "__main__":
    main()