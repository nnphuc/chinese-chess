"""
Puzzle solver for Cờ Thế (Chinese Chess endgame puzzles).
Uses minimax with alpha-beta pruning to find forced mate sequences.
"""

from __future__ import annotations

import time

from loguru import logger

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
        # Current player has no moves and is in check → they lose.
        # Use +depth so shorter mates (more remaining depth) score higher.
        return (-MATE_SCORE - depth, []) if maximizing else (MATE_SCORE + depth, [])

    if board.is_stalemate():
        # In Chinese chess, stalemate (困毙) is a loss for the player with no moves.
        # Same scoring convention as checkmate.
        return (-MATE_SCORE - depth, []) if maximizing else (MATE_SCORE + depth, [])

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
    logger.info("Solving | turn={} | depth={}", board.turn.name, max_depth)

    if max_depth == 0:
        score = evaluate(board)
        logger.debug("Depth 0 evaluation: {}", score)
        return {"score": score, "pv": [], "mate_in": None}

    maximizing = board.turn == Color.RED

    if board.is_checkmate():
        score = (-MATE_SCORE - max_depth) if maximizing else (MATE_SCORE + max_depth)
        logger.info("Already in checkmate | score={}", score)
        return {"score": score, "pv": [], "mate_in": 0}

    if board.is_stalemate():
        # In Chinese chess, stalemate (困毙) is a loss for the player with no moves
        score = (-MATE_SCORE - max_depth) if maximizing else (MATE_SCORE + max_depth)
        logger.info("Already in stalemate (困毙) | score={}", score)
        return {"score": score, "pv": [], "mate_in": 0}

    moves = board.legal_moves()
    assert len(moves) > 0, "No moves but neither checkmate nor stalemate — legality bug"
    logger.info("Top-level candidates: {} move(s)", len(moves))

    alpha, beta = -INF, INF
    best = -INF if maximizing else INF
    best_line: list[Move] = []
    t0 = time.perf_counter()

    for i, move in enumerate(moves, 1):
        notation = move_to_str(move, board)
        child = board.apply_move(move)
        score, line = alphabeta(child, max_depth - 1, alpha, beta, not maximizing)
        logger.debug("[{}/{}] {} → score={}", i, len(moves), notation, score)

        if (maximizing and score > best) or (not maximizing and score < best):
            best = score
            best_line = [move] + line
            logger.debug("  New best: {}", best)

        if maximizing:
            alpha = max(alpha, best)
        else:
            beta = min(beta, best)
        if beta <= alpha:
            logger.debug("  Pruned after {}/{} moves", i, len(moves))
            break

    elapsed = time.perf_counter() - t0
    result: SolveResult = {"score": best, "pv": best_line, "mate_in": None}

    if abs(best) >= MATE_SCORE:
        plies = max_depth - (abs(best) - MATE_SCORE)
        result["mate_in"] = (plies + 1) // 2

    if result["mate_in"] is not None:
        logger.info("Result | Mate in {} | {:.3f}s", result["mate_in"], elapsed)
    else:
        logger.info("Result | score={} | pv={} ply | {:.3f}s", best, len(best_line), elapsed)

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
