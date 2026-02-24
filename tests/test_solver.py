"""Basic smoke tests for the cờ thế solver."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chinese_chess import Board, Color, solve
from src.chinese_chess.board import encode
from src.chinese_chess.pieces import PieceType


def _mate_in_1_board() -> Board:
    grid = [[0] * 9 for _ in range(10)]
    grid[0][3] = encode(Color.BLACK, PieceType.KING)
    grid[1][3] = encode(Color.RED, PieceType.CHARIOT)
    grid[9][3] = encode(Color.RED, PieceType.KING)
    return Board.from_array(grid, turn=Color.RED)


def test_start_position_legal_moves() -> None:
    board = Board.start_position()
    moves = board.legal_moves()
    assert len(moves) > 0


def test_mate_in_1_detected() -> None:
    board = _mate_in_1_board()
    result = solve(board, max_depth=3)
    assert result["mate_in"] == 1


def test_mate_in_1_pv_length() -> None:
    board = _mate_in_1_board()
    result = solve(board, max_depth=3)
    assert len(result["pv"]) == 1


def test_board_copy_independence() -> None:
    board = Board.start_position()
    copy = board.copy()
    copy.grid[0][0] = 0
    assert board.grid[0][0] != 0


# --- Bug fix regression tests ---


def test_flying_kings_two_rows_apart_open_file_is_check() -> None:
    """
    Bug fix: the old code used np.all on an empty NumPy slice (vacuous truth = True).
    Verify the fix doesn't over-correct: kings 2 rows apart on the same column with
    exactly one empty square between them MUST still trigger flying-kings check.
    """
    grid = [[0] * 9 for _ in range(10)]
    # BLACK king row 0 col 4, RED king row 2 col 4 — one empty square between them.
    grid[0][4] = encode(Color.BLACK, PieceType.KING)
    grid[2][4] = encode(Color.RED, PieceType.KING)
    board = Board.from_array(grid, turn=Color.RED)
    assert board._is_in_check(Color.RED)


def test_flying_kings_open_file_is_check() -> None:
    """
    Flying-kings rule MUST fire when kings share a column with no pieces between.
    """
    grid = [[0] * 9 for _ in range(10)]
    grid[0][4] = encode(Color.BLACK, PieceType.KING)
    grid[9][4] = encode(Color.RED, PieceType.KING)
    board = Board.from_array(grid, turn=Color.RED)
    assert board._is_in_check(Color.RED)


def test_flying_kings_blocked_not_check() -> None:
    """
    Flying-kings rule must NOT fire when a piece blocks the file between kings.
    """
    grid = [[0] * 9 for _ in range(10)]
    grid[0][4] = encode(Color.BLACK, PieceType.KING)
    grid[5][4] = encode(Color.RED, PieceType.CHARIOT)  # blocker
    grid[9][4] = encode(Color.RED, PieceType.KING)
    board = Board.from_array(grid, turn=Color.RED)
    assert not board._is_in_check(Color.RED)


def test_mate_in_2_correct_count() -> None:
    """
    Bug fix: mate_in formula must return the correct move count, not a
    value derived from the inverted depth adjustment.
    Construct a forced mate-in-2: RED chariot gives check, BLACK king forced
    to move, second chariot mates.
    """
    grid = [[0] * 9 for _ in range(10)]
    # BLACK king cornered at (0,3) with only one escape
    grid[0][3] = encode(Color.BLACK, PieceType.KING)
    # RED chariot at (2,3) — gives check after moving to (1,3)
    grid[2][3] = encode(Color.RED, PieceType.CHARIOT)
    # Second RED chariot at (1,4) — covers the escape after king moves
    grid[1][4] = encode(Color.RED, PieceType.CHARIOT)
    grid[9][3] = encode(Color.RED, PieceType.KING)
    board = Board.from_array(grid, turn=Color.RED)
    result = solve(board, max_depth=5)
    # Should find a forced mate (mate_in >= 1)
    assert result["mate_in"] is not None
    assert isinstance(result["mate_in"], int)
    assert result["mate_in"] >= 1
