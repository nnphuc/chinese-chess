"""
Chinese Chess (Cờ Thế) Solver — Streamlit Web App

Run with: uv run streamlit run app.py
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

import streamlit as st
import streamlit.elements.image as st_image
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from streamlit_drawable_canvas import st_canvas  # type: ignore[import-not-found]

try:
    # Streamlit >= 1.54 uses image_to_url(image, layout_config, ...), but
    # streamlit-drawable-canvas calls the older signature
    # image_to_url(image, width, ...). Provide an adapter on st_image.
    from streamlit.elements.image import LayoutConfig
    from streamlit.elements.lib.image_utils import image_to_url as _new_image_to_url

    def _compat_image_to_url(
        image: object,
        width_or_layout: object,
        clamp: bool,
        channels: str,
        output_format: str,
        image_id: str,
    ) -> str:
        layout = (
            width_or_layout
            if hasattr(width_or_layout, "width")
            else LayoutConfig(width=cast(int | str | None, width_or_layout))
        )
        return _new_image_to_url(image, layout, clamp, channels, output_format, image_id)

    st_image.image_to_url = _compat_image_to_url  # type: ignore[attr-defined]
except Exception:
    # Best effort compatibility shim; continue and let Streamlit report errors if any.
    pass

from src.chinese_chess import Board, Color, decode, encode, solve
from src.chinese_chess.board import Move
from src.chinese_chess.pieces import PIECE_SYMBOLS, PieceType
from src.chinese_chess.solver import move_to_str

# ── Visual constants ───────────────────────────────────────────────────────────
WOOD = "#F0C070"
WOOD_PALACE = "#E0A840"
WOOD_HIGHLIGHT = "#FFE066"
BORDER_COLOR = "#8B6914"
RED_FG = "#CC0000"
BLACK_FG = "#111111"
CANVAS_W = 620
CANVAS_H = 650
CANVAS_MARGIN_X = 58
CANVAS_MARGIN_Y = 58
CANVAS_DX = 63
CANVAS_DY = 59
CANVAS_PIECE_R = 22

PIECE_ORDER: list[PieceType] = [
    PieceType.KING,
    PieceType.ADVISOR,
    PieceType.ELEPHANT,
    PieceType.HORSE,
    PieceType.CHARIOT,
    PieceType.CANNON,
    PieceType.PAWN,
]

PIECE_LABEL: dict[PieceType, str] = {
    PieceType.KING: "King",
    PieceType.ADVISOR: "Advisor",
    PieceType.ELEPHANT: "Elephant",
    PieceType.HORSE: "Horse",
    PieceType.CHARIOT: "Chariot",
    PieceType.CANNON: "Cannon",
    PieceType.PAWN: "Pawn",
}

# ── Session state ──────────────────────────────────────────────────────────────


def init_state() -> None:
    defaults: dict[str, object] = {
        "board_grid": [[0] * 9 for _ in range(10)],
        "turn": "RED",
        "mode": "setup",
        "pv": [],
        "step": 0,
        "selected": None,
        "result": None,
        "initial_grid": None,
        "depth": 5,
        "setup_canvas_nonce": 0,
        "upload_nonce": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Board helpers ──────────────────────────────────────────────────────────────


def is_palace(r: int, c: int) -> bool:
    return 3 <= c <= 5 and (r <= 2 or r >= 7)


def get_board_at_step(step: int) -> Board:
    initial: list[list[int]] = cast(
        list[list[int]],
        st.session_state.initial_grid or [[0] * 9 for _ in range(10)],
    )
    turn = Color.RED if st.session_state.turn == "RED" else Color.BLACK
    pv: list[Move] = cast(list[Move], st.session_state.pv)
    board = Board.from_array(initial, turn=turn)
    for i in range(min(step, len(pv))):
        board = board.apply_move(pv[i])
    return board


# ── HTML board table builder (shared by both modes) ───────────────────────────


def _piece_circle(val: int, bg: str, inherit_bg: bool = False) -> str:
    if val == 0:
        return f"<span style='color:{BORDER_COLOR};font-size:16px;'>·</span>"
    color, pt = decode(val)
    sym = PIECE_SYMBOLS[(color, pt)]
    fg = RED_FG if color == Color.RED else BLACK_FG
    piece_bg = "inherit" if inherit_bg else bg
    return (
        f"<div style='width:44px;height:44px;border-radius:50%;"
        f"border:2.5px solid {fg};background:{piece_bg};"
        f"display:inline-flex;align-items:center;justify-content:center;"
        f"font-size:21px;font-weight:bold;color:{fg};line-height:1;'>{sym}</div>"
    )


def _board_table(
    grid: list[list[int]],
    last_move: Move | None = None,
) -> str:
    """Return a <table> HTML string for the board."""
    highlight: set[tuple[int, int]] = set()
    if last_move:
        (r1, c1), (r2, c2) = last_move
        highlight = {(r1, c1), (r2, c2)}

    hdr = (
        "width:56px;height:22px;text-align:center;font-size:11px;color:#6B4C11;font-weight:normal;"
    )
    rank_s = "width:26px;text-align:center;font-size:11px;color:#6B4C11;"
    table_s = (
        f"border-collapse:collapse;background:{WOOD};"
        "font-family:'Noto Serif SC','Noto Serif CJK SC',STSong,serif;margin:0 auto;"
    )

    rows: list[str] = []
    col_hdrs = "".join(f"<th style='{hdr}'>{ch}</th>" for ch in "abcdefghi")
    rows.append(f"<tr><th style='width:26px;'></th>{col_hdrs}<th style='width:26px;'></th></tr>")

    for r in range(10):
        rank = 9 - r
        cells = [f"<td style='{rank_s}'>{rank}</td>"]

        for c in range(9):
            val = grid[r][c]
            if (r, c) in highlight:
                bg = WOOD_HIGHLIGHT
            elif is_palace(r, c):
                bg = WOOD_PALACE
            else:
                bg = WOOD

            td_s = (
                f"width:56px;height:54px;text-align:center;vertical-align:middle;"
                f"border:1px solid {BORDER_COLOR};background:{bg};"
            )
            cells.append(f"<td style='{td_s}'>{_piece_circle(val, bg)}</td>")

        cells.append(f"<td style='{rank_s}'>{rank}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

        if r == 4:
            river = (
                f"<td colspan='9' style='height:26px;background:{WOOD};"
                f"text-align:center;font-size:13px;color:{BORDER_COLOR};"
                f"letter-spacing:6px;"
                f"border-left:1px solid {BORDER_COLOR};border-right:1px solid {BORDER_COLOR};'>"
                "楚河&nbsp;&nbsp;&nbsp;&nbsp;漢界</td>"
            )
            rows.append(f"<tr><td></td>{river}<td></td></tr>")

    return f"<table style='{table_s}'>{''.join(rows)}</table>"


def board_to_html(grid: list[list[int]], last_move: Move | None = None) -> str:
    """Read-only board for st.markdown()."""
    return _board_table(grid, last_move)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _setup_board_image(grid: list[list[int]]) -> Image.Image:
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), WOOD)
    draw = ImageDraw.Draw(img)

    x0 = CANVAS_MARGIN_X
    y0 = CANVAS_MARGIN_Y
    dx = CANVAS_DX
    dy = CANVAS_DY

    for r in range(10):
        y = y0 + r * dy
        draw.line([(x0, y), (x0 + 8 * dx, y)], fill=BORDER_COLOR, width=2)

    for c in range(9):
        x = x0 + c * dx
        if c in (0, 8):
            draw.line([(x, y0), (x, y0 + 9 * dy)], fill=BORDER_COLOR, width=2)
        else:
            draw.line([(x, y0), (x, y0 + 4 * dy)], fill=BORDER_COLOR, width=2)
            draw.line([(x, y0 + 5 * dy), (x, y0 + 9 * dy)], fill=BORDER_COLOR, width=2)

    draw.line([(x0 + 3 * dx, y0), (x0 + 5 * dx, y0 + 2 * dy)], fill=BORDER_COLOR, width=2)
    draw.line([(x0 + 5 * dx, y0), (x0 + 3 * dx, y0 + 2 * dy)], fill=BORDER_COLOR, width=2)
    draw.line([(x0 + 3 * dx, y0 + 7 * dy), (x0 + 5 * dx, y0 + 9 * dy)], fill=BORDER_COLOR, width=2)
    draw.line([(x0 + 5 * dx, y0 + 7 * dy), (x0 + 3 * dx, y0 + 9 * dy)], fill=BORDER_COLOR, width=2)

    label_font = _font(20)
    coord_font = _font(16)
    piece_font = _font(28)

    draw.text(
        (x0 + 4 * dx, y0 + int(4.5 * dy)),
        "楚河        漢界",
        fill=BORDER_COLOR,
        font=label_font,
        anchor="mm",
    )

    for c, ch in enumerate("abcdefghi"):
        draw.text((x0 + c * dx, y0 - 26), ch, fill=BORDER_COLOR, font=coord_font, anchor="mm")
    for r in range(10):
        y = y0 + r * dy
        rank = str(9 - r)
        draw.text((x0 - 28, y), rank, fill=BORDER_COLOR, font=coord_font, anchor="mm")
        draw.text((x0 + 8 * dx + 28, y), rank, fill=BORDER_COLOR, font=coord_font, anchor="mm")

    for r in range(10):
        for c in range(9):
            val = grid[r][c]
            if val == 0:
                continue
            color, pt = decode(val)
            fg = RED_FG if color == Color.RED else BLACK_FG
            sym = PIECE_SYMBOLS[(color, pt)]
            cx = x0 + c * dx
            cy = y0 + r * dy
            draw.ellipse(
                [
                    (cx - CANVAS_PIECE_R, cy - CANVAS_PIECE_R),
                    (cx + CANVAS_PIECE_R, cy + CANVAS_PIECE_R),
                ],
                fill=WOOD,
                outline=fg,
                width=3,
            )
            draw.text((cx, cy + 1), sym, fill=fg, font=piece_font, anchor="mm")

    return img


def _canvas_click_to_cell(x: float, y: float) -> tuple[int, int] | None:
    c = round((x - CANVAS_MARGIN_X) / CANVAS_DX)
    r = round((y - CANVAS_MARGIN_Y) / CANVAS_DY)
    if 0 <= r < 10 and 0 <= c < 9:
        return r, c
    return None


def render_setup_canvas(grid: list[list[int]]) -> None:
    img = _setup_board_image(grid)
    nonce = cast(int, st.session_state.setup_canvas_nonce)
    result = st_canvas(
        background_image=img,
        update_streamlit=True,
        height=CANVAS_H,
        width=CANVAS_W,
        drawing_mode="point",
        fill_color="rgba(0,0,0,0)",
        stroke_width=1,
        stroke_color="rgba(0,0,0,0)",
        point_display_radius=0.1,
        display_toolbar=False,
        key=f"setup_canvas_{nonce}",
    )

    json_data = cast(dict[str, object] | None, result.json_data)
    if not json_data:
        return
    objects = cast(list[dict[str, object]], json_data.get("objects") or [])
    if not objects:
        return

    obj = objects[-1]
    x = float(obj.get("left", 0.0))
    y = float(obj.get("top", 0.0))
    radius = obj.get("radius")
    if isinstance(radius, int | float):
        x += float(radius)
        y += float(radius)

    cell = _canvas_click_to_cell(x, y)
    if cell is None:
        return  # outside board — skip rerun

    if cast(str | None, st.session_state.selected) is None:
        return  # nothing selected — skip rerun

    r, c = cell
    handle_cell_click(r, c)
    st.session_state.setup_canvas_nonce = nonce + 1
    st.rerun()


# ── Cell click handler ─────────────────────────────────────────────────────────


def handle_cell_click(r: int, c: int) -> None:
    selected: str | None = cast(str | None, st.session_state.selected)
    grid: list[list[int]] = cast(list[list[int]], st.session_state.board_grid)

    if selected is None:
        return
    if selected == "ERASE":
        grid[r][c] = 0
    else:
        color_str, pt_str = selected.split("_", 1)
        color = Color.RED if color_str == "RED" else Color.BLACK
        grid[r][c] = encode(color, PieceType[pt_str])

    st.session_state.board_grid = grid


# ── Predefined puzzles ────────────────────────────────────────────────────────


def _load_start() -> None:
    st.session_state.board_grid = Board.start_position().grid.tolist()
    st.session_state.turn = "RED"


def _load_mate1() -> None:
    grid = [[0] * 9 for _ in range(10)]
    grid[0][4] = encode(Color.BLACK, PieceType.KING)
    grid[0][3] = encode(Color.BLACK, PieceType.ADVISOR)
    grid[0][5] = encode(Color.BLACK, PieceType.ADVISOR)
    grid[2][4] = encode(Color.RED, PieceType.CHARIOT)
    grid[9][4] = encode(Color.RED, PieceType.KING)
    st.session_state.board_grid = grid
    st.session_state.turn = "RED"


def _load_mate3() -> None:
    grid = [[0] * 9 for _ in range(10)]
    grid[0][4] = encode(Color.BLACK, PieceType.KING)
    grid[3][5] = encode(Color.RED, PieceType.CHARIOT)
    grid[5][3] = encode(Color.RED, PieceType.CHARIOT)
    grid[9][3] = encode(Color.RED, PieceType.KING)
    st.session_state.board_grid = grid
    st.session_state.turn = "RED"


PUZZLES: dict[str, Callable[[], None] | None] = {
    "— Select a puzzle —": None,
    "Start Position": _load_start,
    "Mate-in-1 (Chariot)": _load_mate1,
    "Mate-in-3 (Two Chariots)": _load_mate3,
}


# ── Setup board ───────────────────────────────────────────────────────────────


def _render_setup_board() -> None:
    """Info message + canvas for piece placement."""
    grid = cast(list[list[int]], st.session_state.board_grid)
    sel = cast(str | None, st.session_state.selected)

    if sel == "ERASE":
        st.info("✕ Erase mode — click a cell to remove its piece")
    elif sel:
        color_str, pt_str = sel.split("_", 1)
        pt = PieceType[pt_str]
        color = Color.RED if color_str == "RED" else Color.BLACK
        sym = PIECE_SYMBOLS[(color, pt)]
        st.info(f"Placing: {color_str} {sym} ({PIECE_LABEL[pt]}) — click a board cell")
    else:
        st.caption("Select a piece from the sidebar, then click a board cell to place it.")

    render_setup_canvas(grid)


# ── Main app ──────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="Cờ Thế Solver", page_icon="♟", layout="wide")

    # Style sidebar palette buttons
    st.markdown(
        """<style>
        section[data-testid="stSidebar"] .stButton button {
            min-height: 38px;
            padding: 4px !important;
            font-size: 18px !important;
            font-family: 'Noto Serif SC', STSong, serif !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    init_state()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("♟ Cờ Thế Solver")
        st.caption("Chinese Chess Endgame Puzzle Solver")
        st.divider()

        st.subheader("Load Puzzle")
        puzzle_choice = st.selectbox(
            "puzzle", list(PUZZLES.keys()), index=0, label_visibility="collapsed"
        )
        if st.button(
            "Load ▼",
            disabled=(puzzle_choice == "— Select a puzzle —"),
            use_container_width=True,
        ):
            fn = PUZZLES[puzzle_choice]
            if callable(fn):
                fn()
            st.session_state.mode = "setup"
            st.session_state.pv = []
            st.session_state.step = 0
            st.session_state.result = None
            st.session_state.selected = None
            st.session_state.setup_canvas_nonce = cast(int, st.session_state.setup_canvas_nonce) + 1
            st.rerun()

        st.divider()

        st.subheader("Save / Load Game")

        game_json = json.dumps(
            {
                "grid": cast(list[list[int]], st.session_state.board_grid),
                "turn": cast(str, st.session_state.turn),
            },
            indent=2,
        )
        st.download_button(
            label="💾 Save Game",
            data=game_json,
            file_name="chinese_chess_game.json",
            mime="application/json",
            use_container_width=True,
        )

        upload_nonce = cast(int, st.session_state.upload_nonce)
        loaded_file = st.file_uploader(
            "Load game file (JSON)",
            type=["json"],
            label_visibility="collapsed",
            key=f"game_upload_{upload_nonce}",
        )
        if loaded_file is not None:
            try:
                raw = json.loads(loaded_file.read())
                grid = raw["grid"]
                turn = raw["turn"]
                if (
                    isinstance(grid, list)
                    and len(grid) == 10
                    and all(isinstance(row, list) and len(row) == 9 for row in grid)
                    and isinstance(turn, str)
                    and turn in ("RED", "BLACK")
                ):
                    st.session_state.board_grid = grid
                    st.session_state.turn = turn
                    st.session_state.mode = "setup"
                    st.session_state.pv = []
                    st.session_state.step = 0
                    st.session_state.result = None
                    st.session_state.selected = None
                    st.session_state.upload_nonce = upload_nonce + 1
                    st.session_state.setup_canvas_nonce = (
                        cast(int, st.session_state.setup_canvas_nonce) + 1
                    )
                    logger.info("Game loaded | turn={}", turn)
                    st.rerun()
                else:
                    st.error("Invalid game file format.")
            except Exception as e:
                st.error(f"Failed to load game: {e}")
                logger.error("Failed to load game file: {}", e)

        st.divider()

        if st.session_state.mode == "setup":
            st.subheader("Place Pieces")
            turn_choice = st.radio(
                "Side to move",
                ["RED", "BLACK"],
                index=0 if st.session_state.turn == "RED" else 1,
                horizontal=True,
            )
            st.session_state.turn = turn_choice

            st.caption("Select a piece, then click a board cell:")

            st.write("🔴 RED pieces:")
            red_cols = st.columns(7)
            for i, pt in enumerate(PIECE_ORDER):
                key = f"RED_{pt.name}"
                sym = PIECE_SYMBOLS[(Color.RED, pt)]
                is_sel = st.session_state.selected == key
                if red_cols[i].button(
                    sym,
                    key=f"pal_r_{pt.name}",
                    type="primary" if is_sel else "secondary",
                    help=f"RED {PIECE_LABEL[pt]}",
                ):
                    st.session_state.selected = None if is_sel else key
                    st.rerun()

            st.write("⚫ BLACK pieces:")
            blk_cols = st.columns(7)
            for i, pt in enumerate(PIECE_ORDER):
                key = f"BLACK_{pt.name}"
                sym = PIECE_SYMBOLS[(Color.BLACK, pt)]
                is_sel = st.session_state.selected == key
                if blk_cols[i].button(
                    sym,
                    key=f"pal_b_{pt.name}",
                    type="primary" if is_sel else "secondary",
                    help=f"BLACK {PIECE_LABEL[pt]}",
                ):
                    st.session_state.selected = None if is_sel else key
                    st.rerun()

            erase_col, clear_col = st.columns(2)
            is_erase = st.session_state.selected == "ERASE"
            if erase_col.button(
                "✕ Erase",
                type="primary" if is_erase else "secondary",
                use_container_width=True,
            ):
                st.session_state.selected = None if is_erase else "ERASE"
                st.rerun()
            if clear_col.button("🗑 Clear", use_container_width=True):
                st.session_state.board_grid = [[0] * 9 for _ in range(10)]
                st.session_state.selected = None
                st.session_state.setup_canvas_nonce = (
                    cast(int, st.session_state.setup_canvas_nonce) + 1
                )
                st.rerun()

            st.divider()
            st.subheader("Solve")
            depth = st.slider(
                "Search depth (ply)",
                min_value=3,
                max_value=7,
                value=cast(int, st.session_state.depth),
            )
            st.session_state.depth = depth

            if st.button("▶ Solve", type="primary", use_container_width=True):
                grid_solve: list[list[int]] = cast(list[list[int]], st.session_state.board_grid)
                turn_color = Color.RED if st.session_state.turn == "RED" else Color.BLACK
                board = Board.from_array(grid_solve, turn=turn_color)
                with st.spinner("Searching for best line…"):
                    result = solve(board, max_depth=depth)
                st.session_state.result = result
                st.session_state.pv = cast(list[Move], result["pv"])
                st.session_state.initial_grid = [row[:] for row in grid_solve]
                st.session_state.step = 0
                st.session_state.mode = "replay"
                st.rerun()

        else:  # replay sidebar
            st.subheader("Solution")
            stored_result: dict[str, object] | None = cast(
                dict[str, object] | None, st.session_state.result
            )
            if stored_result:
                mate_in = stored_result.get("mate_in")
                pv_list: list[Move] = cast(list[Move], st.session_state.pv)
                if mate_in is not None:
                    st.success(f"★ Forced mate in {mate_in} move(s)!")
                elif pv_list:
                    st.info(f"Best line: {len(pv_list)} ply")
                else:
                    st.warning("No solution found.")

            if st.button("← Back to Setup", use_container_width=True):
                st.session_state.mode = "setup"
                st.session_state.step = 0
                st.session_state.selected = None
                st.session_state.setup_canvas_nonce = (
                    cast(int, st.session_state.setup_canvas_nonce) + 1
                )
                st.rerun()

    # ── Main area ─────────────────────────────────────────────────────────────
    mode: str = cast(str, st.session_state.mode)

    if mode == "setup":
        st.header("Position Setup")
        _render_setup_board()

    else:  # replay mode
        pv: list[Move] = cast(list[Move], st.session_state.pv)
        step: int = cast(int, st.session_state.step)
        total = len(pv)

        st.header("Solution Replay")

        cur_board = get_board_at_step(step)
        grid_now = cur_board.grid.tolist()
        last_move: Move | None = pv[step - 1] if step > 0 else None

        if step == 0:
            st.markdown("**Initial position** — press Next to step through the solution")
        else:
            prev_board = get_board_at_step(step - 1)
            notation = move_to_str(pv[step - 1], prev_board)
            side = "RED" if prev_board.turn == Color.RED else "BLACK"
            st.markdown(f"**Move {step}** ({side}): `{notation}`")

        if cur_board.is_checkmate():
            winner = "RED" if cur_board.turn == Color.BLACK else "BLACK"
            st.error(f"★ CHECKMATE — {winner} wins! ★")
        elif cur_board.is_stalemate():
            st.warning("— Stalemate —")

        st.markdown(board_to_html(grid_now, last_move), unsafe_allow_html=True)

        st.markdown("")
        prev_col, info_col, next_col = st.columns([2, 3, 2])
        if prev_col.button("← Prev", disabled=(step == 0), use_container_width=True):
            st.session_state.step = step - 1
            st.rerun()
        info_col.markdown(
            f"<div style='text-align:center;padding:8px 0;font-size:14px;'>"
            f"Step {step} / {total}</div>",
            unsafe_allow_html=True,
        )
        if next_col.button("Next →", disabled=(step >= total), use_container_width=True):
            st.session_state.step = step + 1
            st.rerun()


if __name__ == "__main__":
    main()
