"""
Puzzle solver for Cờ Thế (Chinese Chess endgame puzzles).
Uses minimax with alpha-beta pruning to find forced mate sequences.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from .board import Board, Move, decode
from .pieces import PIECE_SYMBOLS, Color, PieceType

INF = 100_000
MATE_SCORE = 90_000
# Any absolute score above this threshold is treated as a mate score.
# Max material without king ≈ 4 800; mate scores start at MATE_SCORE - max_plies.
_MATE_BOUND = MATE_SCORE - 500

# Shared piece values used by both evaluate() and move ordering.
_PIECE_VALUES: dict[PieceType, int] = {
    PieceType.KING: 100_000,
    PieceType.CHARIOT: 900,
    PieceType.CANNON: 450,
    PieceType.HORSE: 400,
    PieceType.ADVISOR: 200,
    PieceType.ELEPHANT: 200,
    PieceType.PAWN: 100,
}

# Vectorised evaluation table: _EVAL_TABLE[val+7] = RED's score contribution
# for an encoded piece value `val` (range -7..+7, 0 = empty).
_EVAL_TABLE: np.ndarray = np.zeros(15, dtype=np.int32)
for _v in range(1, 8):  # RED pieces (val > 0)
    _EVAL_TABLE[_v + 7] = _PIECE_VALUES[PieceType(_v)]
for _v in range(-7, 0):  # BLACK pieces (val < 0)
    _EVAL_TABLE[_v + 7] = -_PIECE_VALUES[PieceType(-_v)]

# Move-ordering tables.
# _Killers: per-depth slots; each slot holds up to 2 quiet moves that caused
#           a β-cutoff at that remaining-depth level.
# _History: accumulated bonus for (r1,c1,r2,c2) pairs that caused cutoffs,
#           weighted by depth² so deep cutoffs matter more.
_Killers = list[list[Move | None]]
_History = dict[tuple[int, int, int, int], int]

# Transposition table — cache previously searched positions keyed by Zobrist hash.
_TT_EXACT = 0  # score is exact
_TT_LOWER = 1  # fail-high (β-cutoff): score is a lower bound
_TT_UPPER = 2  # fail-low  (α-cutoff): score is an upper bound


@dataclass
class _TTEntry:
    depth: int
    score: int
    flag: int  # one of _TT_EXACT / _TT_LOWER / _TT_UPPER
    best_move: Move | None


_TT = dict[int, _TTEntry]


def _tt_store_score(score: int, ply: int) -> int:
    """Normalise a mate score for TT storage by removing the ply distance.

    Mate scores are `±(MATE_SCORE - ply)`.  To make them position-relative
    (independent of *where* in the tree the position was reached), we convert
    them to `±MATE_SCORE` before storing so they can be correctly denormalised
    when retrieved at a potentially different ply.
    """
    if score > _MATE_BOUND:
        return score + ply  # (MATE_SCORE - ply) + ply  → MATE_SCORE
    if score < -_MATE_BOUND:
        return score - ply  # (-MATE_SCORE + ply) - ply → -MATE_SCORE
    return score


def _tt_get_score(score: int, ply: int) -> int:
    """Denormalise a TT score by re-adding the current ply distance."""
    if score > _MATE_BOUND:
        return score - ply  # MATE_SCORE - ply
    if score < -_MATE_BOUND:
        return score + ply  # -MATE_SCORE + ply
    return score


SolveResult = dict[str, object]


class _Timeout(Exception):
    """Raised inside alphabeta when the search deadline has passed."""


def _make_killers(size: int) -> _Killers:
    """Allocate a killer table with `size` depth slots, each holding 2 moves."""
    return [[None, None] for _ in range(size)]


def _update_killer(killers: _Killers, depth: int, move: Move) -> None:
    """Record a quiet β-cutoff move; shift old killer to the second slot."""
    if depth >= len(killers):
        return
    slot = killers[depth]
    if move != slot[0]:
        slot[1] = slot[0]
        slot[0] = move


def evaluate(board: Board) -> int:
    """Material evaluation from RED's perspective (vectorised)."""
    return int(_EVAL_TABLE[board.grid.ravel().astype(np.int16) + 7].sum())


def move_to_str(move: Move, board: Board) -> str:
    (r1, c1), (r2, c2) = move
    val = int(board.grid[r1, c1])
    if val == 0:
        return "???"
    color, pt = decode(val)
    sym = PIECE_SYMBOLS[(color, pt)]
    cols = "abcdefghi"
    return f"{sym}{cols[c1]}{9 - r1}-{cols[c2]}{9 - r2}"


def _move_order_score(
    move: Move,
    board: Board,
    depth: int,
    killers: _Killers,
    history: _History,
    pv_move: Move | None = None,
    tt_move: Move | None = None,
) -> int:
    """
    Heuristic score for move ordering — higher value = searched first.

    Priority (high → low):
      1. PV move from the previous iterative-deepening iteration  (60 000)
      2. TT move — best move stored in the transposition table    (55 000)
      3. Captures — MVV-LVA: cheap attacker × expensive victim    (40 000+)
      4. Killer moves — quiet β-cutoff moves at this depth        (30 000 / 29 000)
      5. Quiet moves that give check                              (20 000)
      6. History heuristic score for remaining quiet moves        (0 … depth²)
    """
    (r1, c1), (r2, c2) = move

    # 1. PV move — must be searched first to get the best window for pruning.
    if move == pv_move:
        return 60_000

    # 2. TT move — best move from a previous search stored at this node.
    if move == tt_move:
        return 55_000

    target = int(board.grid[r2, c2])

    # 3. Captures (MVV-LVA): most valuable victim, least valuable attacker.
    if target != 0:
        _, victim_pt = decode(target)
        _, attacker_pt = decode(int(board.grid[r1, c1]))
        return 40_000 + _PIECE_VALUES[victim_pt] * 10 - _PIECE_VALUES[attacker_pt]

    # 4. Killer moves: quiet moves that caused β-cutoffs at this depth before.
    slot = killers[depth] if depth < len(killers) else [None, None]
    if move == slot[0]:
        return 30_000
    if move == slot[1]:
        return 29_000

    # 5. History heuristic: prefer moves that have historically caused cutoffs.
    return history.get((r1, c1, r2, c2), 0)


def alphabeta(
    board: Board,
    depth: int,
    alpha: int,
    beta: int,
    maximizing: bool,
    deadline: float = float("inf"),
    killers: _Killers | None = None,
    history: _History | None = None,
    pv_hint: list[Move] | None = None,
    tt: _TT | None = None,
    extensions_left: int = 0,
    ply: int = 0,
) -> tuple[int, list[Move]]:
    """Alpha-beta pruning with move ordering, transposition table, and check
    extensions.  Returns (score, pv_line).  Raises _Timeout if deadline passed.

    Move ordering priority at each node:
      PV hint → TT move → captures (MVV-LVA) → killers → checks → history
    Quiet β-cutoff moves update the killer and history tables in-place.
    The TT caches results so transpositions are not re-searched.
    Check extensions: when a move gives check, depth is extended by 1 (up to
    the `extensions_left` budget) so mating lines are searched more deeply.

    Mate scoring uses ±(MATE_SCORE - ply) so the value is independent of
    the remaining depth and unaffected by extensions.  TT entries normalise
    the ply distance on store/retrieve for cross-iteration consistency.
    """
    if time.perf_counter() > deadline:
        raise _Timeout()

    if depth == 0:
        return (evaluate(board), [])

    if killers is None:
        killers = _make_killers(depth + 1)
    if history is None:
        history = {}
    if tt is None:
        tt = {}

    alpha_orig = alpha
    beta_orig = beta

    # --- Transposition table probe -------------------------------------------
    h = board.zobrist_hash
    tt_entry = tt.get(h)
    tt_move: Move | None = None
    if tt_entry is not None:
        tt_move = tt_entry.best_move
        if tt_entry.depth >= depth:
            # Denormalise the stored score back to the current ply distance.
            s = _tt_get_score(tt_entry.score, ply)
            if tt_entry.flag == _TT_EXACT:
                return s, ([tt_move] if tt_move else [])
            elif tt_entry.flag == _TT_LOWER:
                alpha = max(alpha, s)
            elif tt_entry.flag == _TT_UPPER:
                beta = min(beta, s)
            if alpha >= beta:
                return s, ([tt_move] if tt_move else [])
    # -------------------------------------------------------------------------

    pv_move = pv_hint[0] if pv_hint else None

    moves = board.legal_moves()
    if len(moves) == 0:
        # Checkmate or stalemate — in Chinese chess both are a loss for the player to move.
        return (-MATE_SCORE + ply, []) if maximizing else (MATE_SCORE - ply, [])

    # Order: PV → TT move → captures (MVV-LVA) → killers → history.
    moves = sorted(
        moves,
        key=lambda m: _move_order_score(m, board, depth, killers, history, pv_move, tt_move),
        reverse=True,
    )

    best_line: list[Move] = []

    if maximizing:
        best = -INF
        for move in moves:
            (r1, c1), (r2, c2) = move
            next_pv = pv_hint[1:] if (pv_move is not None and move == pv_move and pv_hint) else None
            was_quiet = int(board.grid[r2, c2]) == 0
            captured = board.make_move(move)
            # Check extension: search one ply deeper when the move gives check.
            gives_check = board._is_in_check(board.turn)
            ext = 1 if (gives_check and extensions_left > 0) else 0
            try:
                score, line = alphabeta(
                    board,
                    depth - 1 + ext,
                    alpha,
                    beta,
                    False,
                    deadline,
                    killers,
                    history,
                    next_pv,
                    tt,
                    extensions_left - ext,
                    ply + 1,
                )
            finally:
                board.unmake_move(move, captured)
            if score > best:
                best = score
                best_line = [move] + line
            alpha = max(alpha, best)
            if beta <= alpha:
                if was_quiet:
                    _update_killer(killers, depth, move)
                    history[(r1, c1, r2, c2)] = history.get((r1, c1, r2, c2), 0) + depth * depth
                break
    else:
        best = INF
        for move in moves:
            (r1, c1), (r2, c2) = move
            next_pv = pv_hint[1:] if (pv_move is not None and move == pv_move and pv_hint) else None
            was_quiet = int(board.grid[r2, c2]) == 0
            captured = board.make_move(move)
            gives_check = board._is_in_check(board.turn)
            ext = 1 if (gives_check and extensions_left > 0) else 0
            try:
                score, line = alphabeta(
                    board,
                    depth - 1 + ext,
                    alpha,
                    beta,
                    True,
                    deadline,
                    killers,
                    history,
                    next_pv,
                    tt,
                    extensions_left - ext,
                    ply + 1,
                )
            finally:
                board.unmake_move(move, captured)
            if score < best:
                best = score
                best_line = [move] + line
            beta = min(beta, best)
            if beta <= alpha:
                if was_quiet:
                    _update_killer(killers, depth, move)
                    history[(r1, c1, r2, c2)] = history.get((r1, c1, r2, c2), 0) + depth * depth
                break

    # --- Transposition table store -------------------------------------------
    # Determine the bound type and store only if this result was not truncated.
    flag = _TT_EXACT
    if best <= alpha_orig:
        flag = _TT_UPPER  # failed low: real score ≤ best
    elif best >= beta_orig:
        flag = _TT_LOWER  # failed high: real score ≥ best
    tt[h] = _TTEntry(depth, _tt_store_score(best, ply), flag, best_line[0] if best_line else None)
    # -------------------------------------------------------------------------

    return best, best_line


def solve(
    board: Board,
    max_depth: int = 5,
    deadline: float = float("inf"),
    killers: _Killers | None = None,
    history: _History | None = None,
    pv_hint: list[Move] | None = None,
    tt: _TT | None = None,
) -> SolveResult:
    """
    Solve a cờ thế puzzle. Searches for forced mate up to max_depth plies.
    Returns dict with 'score', 'pv' (principal variation), 'mate_in'.
    Raises _Timeout if deadline is exceeded mid-search.

    `killers`, `history`, and `tt` may be passed in from an outer iterative-
    deepening loop so that ordering knowledge accumulates across iterations.
    `pv_hint` is the PV from the previous iteration; its moves are tried first.
    """
    logger.info("Solving | turn={} | depth={}", board.turn.name, max_depth)

    if max_depth == 0:
        score = evaluate(board)
        logger.debug("Depth 0 evaluation: {}", score)
        return {"score": score, "pv": [], "mate_in": None}

    maximizing = board.turn == Color.RED

    if board.is_checkmate():
        # ply=0 at root
        score = -MATE_SCORE if maximizing else MATE_SCORE
        logger.info("Already in checkmate | score={}", score)
        return {"score": score, "pv": [], "mate_in": 0}

    if board.is_stalemate():
        score = -MATE_SCORE if maximizing else MATE_SCORE
        logger.info("Already in stalemate (困毙) | score={}", score)
        return {"score": score, "pv": [], "mate_in": 0}

    if killers is None:
        killers = _make_killers(max_depth + 2)
    if history is None:
        history = {}
    if tt is None:
        tt = {}

    pv_move = pv_hint[0] if pv_hint else None

    moves = board.legal_moves()
    assert len(moves) > 0, "No moves but neither checkmate nor stalemate — legality bug"

    # Order top-level moves the same way as inner nodes.
    # (No tt_move at the root — the PV hint already covers this.)
    moves = sorted(
        moves,
        key=lambda m: _move_order_score(m, board, max_depth, killers, history, pv_move),
        reverse=True,
    )
    logger.info("Top-level candidates: {} move(s)", len(moves))

    alpha, beta = -INF, INF
    best = -INF if maximizing else INF
    best_line: list[Move] = []
    t0 = time.perf_counter()
    # Allow up to half the base depth in check extensions (conservative cap).
    ext_budget = max_depth // 2

    for i, move in enumerate(moves, 1):
        (r1, c1), (r2, c2) = move
        notation = move_to_str(move, board)
        next_pv = pv_hint[1:] if (pv_move is not None and move == pv_move and pv_hint) else None
        was_quiet = int(board.grid[r2, c2]) == 0
        captured = board.make_move(move)
        gives_check = board._is_in_check(board.turn)
        ext = 1 if (gives_check and ext_budget > 0) else 0
        try:
            score, line = alphabeta(
                board,
                max_depth - 1 + ext,
                alpha,
                beta,
                not maximizing,
                deadline,
                killers,
                history,
                next_pv,
                tt,
                ext_budget - ext,
                1,  # ply=1: one move already made from the root
            )
        finally:
            board.unmake_move(move, captured)
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
            # β-cutoff at root: update killers/history for quiet moves.
            if was_quiet:
                _update_killer(killers, max_depth, move)
                history[(r1, c1, r2, c2)] = history.get((r1, c1, r2, c2), 0) + max_depth * max_depth
            logger.debug("  Pruned after {}/{} moves", i, len(moves))
            break

    elapsed = time.perf_counter() - t0
    result: SolveResult = {"score": best, "pv": best_line, "mate_in": None}

    if abs(best) > _MATE_BOUND:
        # Score = ±(MATE_SCORE - ply_to_checkmate); recover ply from score.
        ply_to_mate = MATE_SCORE - abs(best)
        result["mate_in"] = (ply_to_mate + 1) // 2

    if result["mate_in"] is not None:
        logger.info("Result | Mate in {} | {:.3f}s", result["mate_in"], elapsed)
    else:
        logger.info("Result | score={} | pv={} ply | {:.3f}s", best, len(best_line), elapsed)

    return result


def solve_timed(board: Board, time_limit: float = 10.0) -> SolveResult:
    """
    Iterative deepening search with a time limit.
    Searches depths 3, 5, 7, … until time runs out, returning the result
    from the deepest fully completed search.

    The killer table and history table are shared across iterations so that
    ordering knowledge warm-starts each successive search.  The PV from the
    previous iteration seeds move ordering at every node of the next one.
    """
    logger.info("Solving (time limit: {:.1f}s) | turn={}", time_limit, board.turn.name)
    t0 = time.perf_counter()
    deadline = t0 + time_limit

    maximizing = board.turn == Color.RED
    if board.is_checkmate():
        score = -MATE_SCORE if maximizing else MATE_SCORE
        return {"score": score, "pv": [], "mate_in": 0}
    if board.is_stalemate():
        score = -MATE_SCORE if maximizing else MATE_SCORE
        return {"score": score, "pv": [], "mate_in": 0}

    best_result: SolveResult = {"score": evaluate(board), "pv": [], "mate_in": None}

    # Shared tables — persist and warm across all depth iterations.
    killers: _Killers = _make_killers(202)
    history: _History = {}
    tt: _TT = {}
    prev_pv: list[Move] = []

    for depth in range(3, 200, 2):
        if time.perf_counter() >= deadline:
            logger.info("Time limit reached before starting depth {}", depth)
            break
        try:
            result = solve(
                board,
                max_depth=depth,
                deadline=deadline,
                killers=killers,
                history=history,
                pv_hint=prev_pv,
                tt=tt,
            )
            best_result = result
            elapsed = time.perf_counter() - t0
            logger.info("Depth {} done | mate_in={} | {:.2f}s", depth, result["mate_in"], elapsed)

            # Carry the PV forward to seed the next iteration.
            pv = result.get("pv")
            if isinstance(pv, list) and pv:
                prev_pv = pv

            if result["mate_in"] is not None:
                logger.info("Found forced mate in {}, stopping early", result["mate_in"])
                break
        except _Timeout:
            elapsed = time.perf_counter() - t0
            logger.info("Timeout during depth {} search | {:.2f}s elapsed", depth, elapsed)
            break

    return best_result


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
