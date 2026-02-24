"""
Cờ Thế Solver — Move-by-Move Game Replay
Puzzle: RED forces checkmate in 3 moves with two chariots.
"""

from colorama import Fore, Style, init

from src.chinese_chess import Board, Color, solve
from src.chinese_chess.board import decode, encode
from src.chinese_chess.pieces import PIECE_SYMBOLS, PieceType
from src.chinese_chess.solver import move_to_str

init(autoreset=True)

# ── Color palette ──────────────────────────────────────────────────────────────
RED_PIECE = Fore.RED + Style.BRIGHT
BLACK_PIECE = Fore.CYAN + Style.BRIGHT
BOARD_FG = Fore.WHITE
DIM = Style.DIM
GOLD = Fore.YELLOW + Style.BRIGHT
GREEN = Fore.GREEN + Style.BRIGHT
RESET = Style.RESET_ALL


def colored_board(board: Board) -> str:
    """Return a colored board string."""
    lines = []
    lines.append(BOARD_FG + "   a b c d e f g h i" + RESET)
    lines.append(BOARD_FG + "  ╔═══════════════════╗" + RESET)
    for r in range(10):
        rank = 9 - r
        row_str = BOARD_FG + f"{rank} ║" + RESET
        for c in range(9):
            val = int(board.grid[r, c])
            if val == 0:
                row_str += DIM + " ·" + RESET
            else:
                color, pt = decode(val)
                sym = PIECE_SYMBOLS[(color, pt)]
                color_code = RED_PIECE if color == Color.RED else BLACK_PIECE
                row_str += " " + color_code + sym + RESET
        row_str += BOARD_FG + " ║" + RESET
        lines.append(row_str)
    lines.append(BOARD_FG + "  ╚═══════════════════╝" + RESET)
    turn_label = (
        RED_PIECE + "RED (紅)" if board.turn == Color.RED else BLACK_PIECE + "BLACK (黑)"
    ) + RESET
    lines.append(f"  Turn: {turn_label}")
    return "\n".join(lines)


def puzzle_mate_in_3() -> Board:
    """
    Cờ thế: RED to move, forced mate in 3 (two-chariot scissors).

      . . . . 將 . . . .   rank 10  — BLACK king
      . . . . . . . . .
      . . . . . . . . .
      . . . . . 俥 . . .   rank 7   — RED chariot1
      . . . . . . . . .
      . . . 俥 . . . . .   rank 5   — RED chariot2
      ...
      . . . 帥 . . . . .   rank 1   — RED king
    """
    grid = [[0] * 9 for _ in range(10)]
    grid[0][4] = encode(Color.BLACK, PieceType.KING)
    grid[3][5] = encode(Color.RED, PieceType.CHARIOT)  # chariot1
    grid[5][3] = encode(Color.RED, PieceType.CHARIOT)  # chariot2
    grid[9][3] = encode(Color.RED, PieceType.KING)
    return Board.from_array(grid, turn=Color.RED)


def play_through(board: Board, depth: int = 7) -> None:
    W = 50
    sep = BOARD_FG + "─" * W + RESET
    heading = GOLD + "=" * W + RESET

    print(heading)
    print(GOLD + "    Cờ Thế Solver — Move-by-Move Replay" + RESET)
    print(heading)

    print("\nInitial position:")
    print(colored_board(board))

    print(f"\n{DIM}Solving at depth={depth}…{RESET} ", end="", flush=True)
    result = solve(board, max_depth=depth)
    pv = result["pv"]
    mate_in = result["mate_in"]

    if not pv:
        print(Fore.RED + "No solution found." + RESET)
        return

    assert isinstance(pv, list)

    if mate_in is not None:
        print(GOLD + f"\n  ★  Forced mate in {mate_in} move(s) found!\n" + RESET)
    else:
        print(f"\n{DIM}Best line ({len(pv)} ply):\n{RESET}")

    print(sep)
    b = board
    for i, move in enumerate(pv):
        is_red = b.turn == Color.RED
        side_col = RED_PIECE if is_red else BLACK_PIECE
        side_lbl = "RED  (紅)" if is_red else "BLACK (黑)"
        move_num = i // 2 + 1
        notation = move_to_str(move, b)

        prefix = GREEN + f"Move {move_num}" + RESET if i % 2 == 0 else "      …"
        print(f"{prefix}  [{side_col}{side_lbl}{RESET}]  {Style.BRIGHT}{notation}{RESET}")

        b = b.apply_move(move)
        print(colored_board(b))

        if b.is_checkmate():
            winner = "RED" if b.turn == Color.BLACK else "BLACK"
            w_col = RED_PIECE if winner == "RED" else BLACK_PIECE
            print(GOLD + "\n  ★  CHECKMATE — " + w_col + f"{winner} wins!" + GOLD + "  ★" + RESET)
        elif b.is_stalemate():
            print(DIM + "\n  — Stalemate —" + RESET)

        print(sep)


if __name__ == "__main__":
    play_through(puzzle_mate_in_3(), depth=5)
