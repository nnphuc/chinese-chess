"""
Chinese Chess board representation and move generation.
Board: 10 rows x 9 columns (row 0 = BLACK side top, row 9 = RED side bottom)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .pieces import PIECE_SYMBOLS, Color, PieceType

Move = tuple[tuple[int, int], tuple[int, int]]

# ---------------------------------------------------------------------------
# Pre-encoded piece integer values — avoids encode() calls in hot paths.
# encode(color, pt) = int(color) * int(pt); Color.RED=1, Color.BLACK=-1.
# ---------------------------------------------------------------------------
_RED_KING: int = 1
_RED_ADVISOR: int = 2
_RED_ELEPHANT: int = 3
_RED_HORSE: int = 4
_RED_CHARIOT: int = 5
_RED_CANNON: int = 6
_RED_PAWN: int = 7
_BLACK_KING: int = -1
_BLACK_ADVISOR: int = -2
_BLACK_ELEPHANT: int = -3
_BLACK_HORSE: int = -4
_BLACK_CHARIOT: int = -5
_BLACK_CANNON: int = -6
_BLACK_PAWN: int = -7

# Fast color flip — avoids Color(int(color)*-1) enum construction overhead.
_FLIP: dict[Color, Color] = {Color.RED: Color.BLACK, Color.BLACK: Color.RED}

# ---------------------------------------------------------------------------
# Zobrist hashing — fixed seed so hashes are reproducible across runs.
# Table indexed by [piece_val+7, row*9+col].  Index 7 (val=0) is never used.
# ---------------------------------------------------------------------------
_rng = np.random.default_rng(0xCCECBEEF)
_Z_PIECES: NDArray[np.int64] = _rng.integers(-(2**62), 2**62, size=(15, 90), dtype=np.int64)
_Z_SIDE: int = int(_rng.integers(-(2**62), 2**62, dtype=np.int64))


# Encoding: piece_type * color  (RED=+, BLACK=-)
def encode(color: Color, pt: PieceType) -> int:
    return int(color) * int(pt)


def decode(val: int) -> tuple[Color, PieceType]:
    assert val != 0, "Cannot decode empty square"
    color = Color.RED if val > 0 else Color.BLACK
    return color, PieceType(abs(val))


# Palace bounds
RED_PALACE_ROWS = (7, 9)
BLACK_PALACE_ROWS = (0, 2)
PALACE_COLS = (3, 5)

# River: rows 0-4 = BLACK territory, rows 5-9 = RED territory
RED_SIDE = range(5, 10)
BLACK_SIDE = range(0, 5)


class Board:
    def __init__(self) -> None:
        self.grid: NDArray[np.int8] = np.zeros((10, 9), dtype=np.int8)
        self.turn = Color.RED
        self.hash: int = 0  # Zobrist hash, maintained incrementally
        # King positions — tracked for O(1) check detection; -1 means absent.
        self._rk_r: int = -1  # RED king row
        self._rk_c: int = -1  # RED king col
        self._bk_r: int = -1  # BLACK king row
        self._bk_c: int = -1  # BLACK king col

    def _recompute_hash(self) -> None:
        """Recompute Zobrist hash and king positions from scratch."""
        h = 0
        self._rk_r = self._rk_c = self._bk_r = self._bk_c = -1
        flat = self.grid.ravel()
        for sq in range(90):
            v = int(flat[sq])
            if v != 0:
                h ^= int(_Z_PIECES[v + 7, sq])
                if v == _RED_KING:
                    self._rk_r, self._rk_c = sq // 9, sq % 9
                elif v == _BLACK_KING:
                    self._bk_r, self._bk_c = sq // 9, sq % 9
        if self.turn == Color.BLACK:
            h ^= _Z_SIDE
        self.hash = h

    @classmethod
    def from_array(cls, grid: list[list[int]], turn: Color = Color.RED) -> Board:
        b = cls()
        b.grid = np.array(grid, dtype=np.int8)
        b.turn = turn
        b._recompute_hash()
        return b

    @classmethod
    def start_position(cls) -> Board:
        b = cls()
        g = b.grid
        # BLACK pieces (top)
        g[0] = [
            encode(Color.BLACK, PieceType.CHARIOT),
            encode(Color.BLACK, PieceType.HORSE),
            encode(Color.BLACK, PieceType.ELEPHANT),
            encode(Color.BLACK, PieceType.ADVISOR),
            encode(Color.BLACK, PieceType.KING),
            encode(Color.BLACK, PieceType.ADVISOR),
            encode(Color.BLACK, PieceType.ELEPHANT),
            encode(Color.BLACK, PieceType.HORSE),
            encode(Color.BLACK, PieceType.CHARIOT),
        ]
        g[2, 1] = encode(Color.BLACK, PieceType.CANNON)
        g[2, 7] = encode(Color.BLACK, PieceType.CANNON)
        for c in range(0, 9, 2):
            g[3, c] = encode(Color.BLACK, PieceType.PAWN)
        # RED pieces (bottom)
        g[9] = [
            encode(Color.RED, PieceType.CHARIOT),
            encode(Color.RED, PieceType.HORSE),
            encode(Color.RED, PieceType.ELEPHANT),
            encode(Color.RED, PieceType.ADVISOR),
            encode(Color.RED, PieceType.KING),
            encode(Color.RED, PieceType.ADVISOR),
            encode(Color.RED, PieceType.ELEPHANT),
            encode(Color.RED, PieceType.HORSE),
            encode(Color.RED, PieceType.CHARIOT),
        ]
        g[7, 1] = encode(Color.RED, PieceType.CANNON)
        g[7, 7] = encode(Color.RED, PieceType.CANNON)
        for c in range(0, 9, 2):
            g[6, c] = encode(Color.RED, PieceType.PAWN)
        b._recompute_hash()
        return b

    def copy(self) -> Board:
        b = Board()
        b.grid = self.grid.copy()
        b.turn = self.turn
        b.hash = self.hash
        b._rk_r = self._rk_r
        b._rk_c = self._rk_c
        b._bk_r = self._bk_r
        b._bk_c = self._bk_c
        return b

    @property
    def zobrist_hash(self) -> int:
        """Zobrist hash of this position (maintained incrementally)."""
        return self.hash

    def make_move(self, move: Move) -> int:
        """Apply move in-place. Returns the captured piece value (0 if none)."""
        (r1, c1), (r2, c2) = move
        sq1 = r1 * 9 + c1
        sq2 = r2 * 9 + c2
        piece = int(self.grid[r1, c1])
        captured = int(self.grid[r2, c2])
        # Update Zobrist hash incrementally
        self.hash ^= int(_Z_PIECES[piece + 7, sq1])  # remove piece from src
        if captured != 0:
            self.hash ^= int(_Z_PIECES[captured + 7, sq2])  # remove captured from dst
        self.hash ^= int(_Z_PIECES[piece + 7, sq2])  # place piece at dst
        self.hash ^= _Z_SIDE  # flip side to move
        # Update grid
        self.grid[r2, c2] = piece
        self.grid[r1, c1] = 0
        # Update king position if king moved
        if piece == _RED_KING:
            self._rk_r, self._rk_c = r2, c2
        elif piece == _BLACK_KING:
            self._bk_r, self._bk_c = r2, c2
        # Flip turn
        self.turn = _FLIP[self.turn]
        return captured

    def unmake_move(self, move: Move, captured: int) -> None:
        """Undo a previously applied make_move in-place."""
        (r1, c1), (r2, c2) = move
        sq1 = r1 * 9 + c1
        sq2 = r2 * 9 + c2
        piece = int(self.grid[r2, c2])
        # Reverse Zobrist hash update
        self.hash ^= int(_Z_PIECES[piece + 7, sq2])  # remove piece from dst
        if captured != 0:
            self.hash ^= int(_Z_PIECES[captured + 7, sq2])  # restore captured at dst
        self.hash ^= int(_Z_PIECES[piece + 7, sq1])  # restore piece at src
        self.hash ^= _Z_SIDE  # flip side back
        # Restore grid
        self.grid[r1, c1] = piece
        self.grid[r2, c2] = captured
        # Restore king position
        if piece == _RED_KING:
            self._rk_r, self._rk_c = r1, c1
        elif piece == _BLACK_KING:
            self._bk_r, self._bk_c = r1, c1
        # Flip turn back
        self.turn = _FLIP[self.turn]

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < 10 and 0 <= c < 9

    def _own(self, val: int) -> bool:
        return (int(self.turn) * val) > 0

    def _enemy(self, val: int) -> bool:
        return val != 0 and not self._own(val)

    def legal_moves(self) -> list[Move]:
        moves: list[Move] = []
        g = self.grid
        color = self.turn
        for r in range(10):
            for c in range(9):
                val = int(g[r, c])
                if val == 0 or not self._own(val):
                    continue
                _, pt = decode(val)
                for dst in self._piece_moves(r, c, pt, color):
                    moves.append(((r, c), dst))
        return [m for m in moves if not self._leaves_king_in_check(m)]

    def _piece_moves(self, r: int, c: int, pt: PieceType, color: Color) -> list[tuple[int, int]]:
        g = self.grid
        dests: list[tuple[int, int]] = []

        if pt == PieceType.KING:
            pr0, pr1 = RED_PALACE_ROWS if color == Color.RED else BLACK_PALACE_ROWS
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if pr0 <= nr <= pr1 and PALACE_COLS[0] <= nc <= PALACE_COLS[1]:
                    if not self._own(int(g[nr, nc])):
                        dests.append((nr, nc))

        elif pt == PieceType.ADVISOR:
            pr0, pr1 = RED_PALACE_ROWS if color == Color.RED else BLACK_PALACE_ROWS
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if pr0 <= nr <= pr1 and PALACE_COLS[0] <= nc <= PALACE_COLS[1]:
                    if not self._own(int(g[nr, nc])):
                        dests.append((nr, nc))

        elif pt == PieceType.ELEPHANT:
            side = RED_SIDE if color == Color.RED else BLACK_SIDE
            for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc) and nr in side:
                    mr, mc = r + dr // 2, c + dc // 2
                    if g[mr, mc] == 0 and not self._own(int(g[nr, nc])):
                        dests.append((nr, nc))

        elif pt == PieceType.HORSE:
            for dr, dc, br, bc in [
                (-2, -1, -1, 0),
                (-2, 1, -1, 0),
                (2, -1, 1, 0),
                (2, 1, 1, 0),
                (-1, -2, 0, -1),
                (-1, 2, 0, 1),
                (1, -2, 0, -1),
                (1, 2, 0, 1),
            ]:
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc) and g[r + br, c + bc] == 0:
                    if not self._own(int(g[nr, nc])):
                        dests.append((nr, nc))

        elif pt == PieceType.CHARIOT:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                while self._in_bounds(nr, nc):
                    v = int(g[nr, nc])
                    if v == 0:
                        dests.append((nr, nc))
                    elif self._enemy(v):
                        dests.append((nr, nc))
                        break
                    else:
                        break
                    nr += dr
                    nc += dc

        elif pt == PieceType.CANNON:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                screen = False
                while self._in_bounds(nr, nc):
                    v = int(g[nr, nc])
                    if not screen:
                        if v == 0:
                            dests.append((nr, nc))
                        else:
                            screen = True
                    else:
                        if v != 0:
                            if self._enemy(v):
                                dests.append((nr, nc))
                            break
                    nr += dr
                    nc += dc

        elif pt == PieceType.PAWN:
            forward = -1 if color == Color.RED else 1
            crossed = (r in BLACK_SIDE) if color == Color.RED else (r in RED_SIDE)
            candidates = [(r + forward, c)]
            if crossed:
                candidates += [(r, c - 1), (r, c + 1)]
            for nr, nc in candidates:
                if self._in_bounds(nr, nc) and not self._own(int(g[nr, nc])):
                    dests.append((nr, nc))

        return dests

    def _leaves_king_in_check(self, move: Move) -> bool:
        captured = self.make_move(move)
        in_check = self._is_in_check(_FLIP[self.turn])
        self.unmake_move(move, captured)
        return in_check

    def _is_in_check(self, color: Color) -> bool:
        """Return True if `color`'s king is under attack.

        Uses O(1) king lookup + ray-casting from the king's square instead of
        generating all opponent moves.
        """
        g = self.grid

        # O(1) king position lookup via tracked attributes.
        if color == Color.RED:
            kr, kc = self._rk_r, self._rk_c
            opp_chariot = _BLACK_CHARIOT
            opp_cannon = _BLACK_CANNON
            opp_horse = _BLACK_HORSE
            opp_pawn = _BLACK_PAWN
            opp_king_val = _BLACK_KING
            opp_fwd = 1  # BLACK pawn moves +1 toward RED
            opp_crossed = RED_SIDE  # BLACK pawn attacks sideways in RED territory
        else:
            kr, kc = self._bk_r, self._bk_c
            opp_chariot = _RED_CHARIOT
            opp_cannon = _RED_CANNON
            opp_horse = _RED_HORSE
            opp_pawn = _RED_PAWN
            opp_king_val = _RED_KING
            opp_fwd = -1  # RED pawn moves -1 toward BLACK
            opp_crossed = BLACK_SIDE  # RED pawn attacks sideways in BLACK territory

        if kr == -1:
            return True  # king absent — treat as in check

        # 1. Rook directions: CHARIOT slides / CANNON jumps / flying-kings.
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = kr + dr, kc + dc
            screen = False
            while 0 <= r < 10 and 0 <= c < 9:
                v = int(g[r, c])
                if v == 0:
                    pass
                elif not screen:
                    if v == opp_chariot or v == opp_king_val:
                        return True  # chariot or flying-kings
                    screen = True
                else:
                    if v == opp_cannon:
                        return True  # cannon attack over one screen
                    break
                r += dr
                c += dc

        # 2. HORSE attacks: check 8 possible horse-origin squares.
        for dr, dc, br, bc in (
            (-2, -1, -1, 0),
            (-2, 1, -1, 0),
            (2, -1, 1, 0),
            (2, 1, 1, 0),
            (-1, -2, 0, -1),
            (-1, 2, 0, 1),
            (1, -2, 0, -1),
            (1, 2, 0, 1),
        ):
            hr, hc = kr - dr, kc - dc
            if 0 <= hr < 10 and 0 <= hc < 9:
                blk_r, blk_c = hr + br, hc + bc
                if 0 <= blk_r < 10 and 0 <= blk_c < 9:
                    if int(g[blk_r, blk_c]) == 0 and int(g[hr, hc]) == opp_horse:
                        return True

        # 3. PAWN attacks.
        pr = kr - opp_fwd
        if 0 <= pr < 10 and int(g[pr, kc]) == opp_pawn:
            return True
        if kr in opp_crossed:
            if kc > 0 and int(g[kr, kc - 1]) == opp_pawn:
                return True
            if kc < 8 and int(g[kr, kc + 1]) == opp_pawn:
                return True

        return False

    def _raw_moves(self) -> list[Move]:
        moves: list[Move] = []
        for r in range(10):
            for c in range(9):
                val = int(self.grid[r, c])
                if val == 0 or not self._own(val):
                    continue
                _, pt = decode(val)
                for dst in self._piece_moves(r, c, pt, self.turn):
                    moves.append(((r, c), dst))
        return moves

    def apply_move(self, move: Move) -> Board:
        b = self.copy()
        b.make_move(move)
        return b

    def is_checkmate(self) -> bool:
        return self._is_in_check(self.turn) and len(self.legal_moves()) == 0

    def is_stalemate(self) -> bool:
        return not self._is_in_check(self.turn) and len(self.legal_moves()) == 0

    def display(self) -> str:
        lines = []
        lines.append("   a b c d e f g h i")
        lines.append("  ╔═══════════════════╗")
        for r in range(10):
            row_str = f"{9 - r} ║"
            for c in range(9):
                val = int(self.grid[r, c])
                if val == 0:
                    row_str += " ·"
                else:
                    color, pt = decode(val)
                    row_str += " " + PIECE_SYMBOLS[(color, pt)]
            row_str += " ║"
            lines.append(row_str)
        lines.append("  ╚═══════════════════╝")
        lines.append(f"  Turn: {'RED (紅)' if self.turn == Color.RED else 'BLACK (黑)'}")
        return "\n".join(lines)
