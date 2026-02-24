"""
Chinese Chess pieces definitions.
Pieces: 將/帥(King), 士/仕(Advisor), 象/相(Elephant), 馬/傌(Horse),
        車/俥(Chariot), 炮/砲(Cannon), 卒/兵(Pawn)
"""

from enum import IntEnum


class Color(IntEnum):
    RED = 1
    BLACK = -1


class PieceType(IntEnum):
    KING = 1  # 將/帥
    ADVISOR = 2  # 士/仕
    ELEPHANT = 3  # 象/相
    HORSE = 4  # 馬/傌
    CHARIOT = 5  # 車/俥
    CANNON = 6  # 炮/砲
    PAWN = 7  # 卒/兵


# Board encoding: positive = RED, negative = BLACK, 0 = empty
PIECE_SYMBOLS = {
    (Color.RED, PieceType.KING): "帥",
    (Color.RED, PieceType.ADVISOR): "仕",
    (Color.RED, PieceType.ELEPHANT): "相",
    (Color.RED, PieceType.HORSE): "傌",
    (Color.RED, PieceType.CHARIOT): "俥",
    (Color.RED, PieceType.CANNON): "砲",
    (Color.RED, PieceType.PAWN): "兵",
    (Color.BLACK, PieceType.KING): "將",
    (Color.BLACK, PieceType.ADVISOR): "士",
    (Color.BLACK, PieceType.ELEPHANT): "象",
    (Color.BLACK, PieceType.HORSE): "馬",
    (Color.BLACK, PieceType.CHARIOT): "車",
    (Color.BLACK, PieceType.CANNON): "炮",
    (Color.BLACK, PieceType.PAWN): "卒",
}
