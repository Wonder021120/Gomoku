import pytest

from gomoku.board import Board
from gomoku.game import Game
from gomoku.rules import StandardRule, create_rule


def test_create_standard_rule():
    rule = create_rule("standard")

    assert isinstance(rule, StandardRule)
    assert rule.name == "standard"


def test_create_unknown_rule_raises_error():
    with pytest.raises(ValueError):
        create_rule("unknown")


def test_standard_rule_initial_player_is_black():
    rule = StandardRule()

    assert rule.get_initial_player() == Board.BLACK


def test_standard_rule_allows_empty_cell():
    game = Game(board_size=15, rule_name="standard")
    rule = StandardRule()

    rule.validate_move(game, (7, 7))


def test_standard_rule_rejects_occupied_cell():
    game = Game(board_size=15, rule_name="standard")
    rule = StandardRule()

    game.play_move((7, 7))

    with pytest.raises(ValueError):
        rule.validate_move(game, (7, 7))


def test_standard_rule_rejects_outside_move():
    game = Game(board_size=15, rule_name="standard")
    rule = StandardRule()

    with pytest.raises(ValueError):
        rule.validate_move(game, (15, 0))

    with pytest.raises(ValueError):
        rule.validate_move(game, (-1, 0))


def test_game_uses_standard_rule_by_default():
    game = Game(board_size=15)

    assert game.rule.name == "standard"
    assert game.current_player == Board.BLACK


def test_game_copy_preserves_rule():
    game = Game(board_size=15, rule_name="standard")
    game.play_move((7, 7))

    copied = game.copy()

    assert copied.rule.name == "standard"
    assert copied.rule_name == "standard"
    assert copied.board.grid[7, 7] == Board.BLACK