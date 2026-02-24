"""
Puzzle solver for Cờ Thế (Chinese Chess endgame puzzles).
Uses minimax with alpha-beta pruning to find forced mate sequences.
"""

from __future__ import annotations

from .board import Board, Move, decode
from .pieces import PIECE_SYMBOLS, Color, PieceType

INF = 100_000
MATE_SCORE = 90_000

SolveResult = dict[str, object]


def evaluate(board: Board) -> int:
    """Material evaluation from RED's perspective."""
    PIECE_VALUES: dict[PieceType, int] = {
        PieceType.KING: 100_000,
        PieceType.CHARIOT: 900,
        PieceType.CANNON: 450,
        PieceType.HORSE: 400,
        PieceType.ADVISOR: 200,
        PieceType.ELEPHANT: 200,
        PieceType.PAWN: 100,
    }
    score = 0
    for r in range(10):
        for c in range(9):
            val = int(board.grid[r, c])
            if val == 0:
                continue
            color, pt = decode(val)
            v = PIECE_VALUES[pt]
            score += v if color == Color.RED else -v
    return score


def move_to_str(move: Move, board: Board) -> str:
    (r1, c1), (r2, c2) = move
    val = int(board.grid[r1, c1])
    if val == 0:
        return "???"
    color, pt = decode(val)
    sym = PIECE_SYMBOLS[(color, pt)]
    cols = "abcdefghi"
    return f"{sym}{cols[c1]}{9 - r1}-{cols[c2]}{9 - r2}"


def alphabeta(
    board: Board, depth: int, alpha: int, beta: int, maximizing: bool
) -> tuple[int, list[Move]]:
    """Alpha-beta pruning. Returns (score, pv_line)."""
    if board.is_checkmate():
        # Current player is mated → they lose. Prefer mates found sooner (higher depth remaining).
        return (-MATE_SCORE + depth, []) if maximizing else (MATE_SCORE - depth, [])

    if board.is_stalemate():
        return (0, [])

    if depth == 0:
        return (evaluate(board), [])

    moves = board.legal_moves()
    assert len(moves) > 0, "No moves but neither checkmate nor stalemate — legality bug"
    best_line: list[Move] = []

    if maximizing:
        best = -INF
        for move in moves:
            child = board.apply_move(move)
            score, line = alphabeta(child, depth - 1, alpha, beta, False)
            if score > best:
                best = score
                best_line = [move] + line
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best, best_line
    else:
        best = INF
        for move in moves:
            child = board.apply_move(move)
            score, line = alphabeta(child, depth - 1, alpha, beta, True)
            if score < best:
                best = score
                best_line = [move] + line
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best, best_line


def solve(board: Board, max_depth: int = 5) -> SolveResult:
    """
    Solve a cờ thế puzzle. Searches for forced mate up to max_depth plies.
    Returns dict with 'score', 'pv' (principal variation), 'mate_in'.
    """
    maximizing = board.turn == Color.RED
    score, pv = alphabeta(board, max_depth, -INF, INF, maximizing)

    result: SolveResult = {"score": score, "pv": pv, "mate_in": None}

    # Mate score = MATE_SCORE - depth_at_terminal (always < MATE_SCORE).
    # Threshold: any score that couldn't be from material alone (material max ~10k).
    if abs(score) >= MATE_SCORE - max_depth:
        # depth_at_terminal = MATE_SCORE - abs(score)
        # plies consumed = max_depth - depth_at_terminal
        plies = max_depth - (MATE_SCORE - abs(score))
        result["mate_in"] = (plies + 1) // 2

    return result


def print_solution(board: Board, result: SolveResult) -> None:
    pv = result["pv"]
    mate_in = result["mate_in"]

    if mate_in is not None:
        print(f"Mate in {mate_in} move(s)!")
    else:
        print(f"Best score: {result['score']}")

    if not pv:
        print("No solution found.")
        return

    assert isinstance(pv, list)
    print("\nBest line:")
    b = board
    for i, move in enumerate(pv):
        turn_label = "RED" if b.turn == Color.RED else "BLACK"
        move_num = i // 2 + 1
        notation = move_to_str(move, b)
        if i % 2 == 0:
            print(f"  {move_num}. [{turn_label}] {notation}", end="")
        else:
            print(f"  ... [{turn_label}] {notation}")
        b = b.apply_move(move)
    if len(pv) % 2 == 1:
        print()
