"""
Chinese Chess board representation and move generation.
Board: 10 rows x 9 columns (row 0 = BLACK side top, row 9 = RED side bottom)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .pieces import PIECE_SYMBOLS, Color, PieceType

Move = tuple[tuple[int, int], tuple[int, int]]


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

    @classmethod
    def from_array(cls, grid: list[list[int]], turn: Color = Color.RED) -> Board:
        b = cls()
        b.grid = np.array(grid, dtype=np.int8)
        b.turn = turn
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
        return b

    def copy(self) -> Board:
        b = Board()
        b.grid = self.grid.copy()
        b.turn = self.turn
        return b

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
        b = self.copy()
        (r1, c1), (r2, c2) = move
        b.grid[r2, c2] = b.grid[r1, c1]
        b.grid[r1, c1] = 0
        return b._is_in_check(self.turn)

    def _is_in_check(self, color: Color) -> bool:
        king_val = encode(color, PieceType.KING)
        pos = np.argwhere(self.grid == king_val)
        if len(pos) == 0:
            return True
        kr, kc = int(pos[0][0]), int(pos[0][1])
        # Check flying kings rule
        opp_king_val = encode(Color(int(color) * -1), PieceType.KING)
        opp_pos = np.argwhere(self.grid == opp_king_val)
        if len(opp_pos) > 0:
            okr, okc = int(opp_pos[0][0]), int(opp_pos[0][1])
            if okc == kc and abs(kr - okr) >= 2:
                between = self.grid[min(kr, okr) + 1 : max(kr, okr), kc]
                if np.all(between == 0):
                    return True
        b = self.copy()
        b.turn = Color(int(color) * -1)
        for (_, _), (r2, c2) in b._raw_moves():
            if r2 == kr and c2 == kc:
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
        (r1, c1), (r2, c2) = move
        b.grid[r2, c2] = b.grid[r1, c1]
        b.grid[r1, c1] = 0
        b.turn = Color(int(b.turn) * -1)
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
