# Chinese Chess Puzzle Solver (Cờ Thế / 象棋)

A Chinese Chess (Cờ Tướng / 象棋) endgame puzzle solver with an interactive web UI.
Set up any board position, hit **Solve**, and step through the forced-mate line move by move.

![Python](https://img.shields.io/badge/python-3.12+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **Interactive board setup** — click to place or remove pieces on a canvas
- **Alpha-beta solver** with iterative deepening, transposition table (Zobrist hashing), killer moves, history heuristic, and check extensions
- **Move-by-move replay** with auto-play at configurable speed
- **Save / load positions** as JSON
- **CLI mode** for terminal-based puzzle replay (coloured output)

## Demo

```
★  Forced mate in 3 move(s) found!

Move 1  [RED  (紅)]  俥f7-f10
Move 2  [BLACK (黑)]  將e10-d10
Move 3  [RED  (紅)]  俥d5-d10
★  CHECKMATE — RED wins!
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) **or** pip

## Installation

```bash
git clone https://github.com/nnphuc/chinese-chess.git
cd chinese-chess
uv sync          # install all dependencies
```

## Usage

### Web UI (Streamlit)

```bash
uv run streamlit run app.py
```

Then open http://localhost:8501 in your browser.

**Workflow:**
1. Use the **Setup** tab to place pieces on the board (click a piece in the palette, then click a square)
2. Choose whose turn it is and set the time limit
3. Click **▶ Solve** — the solver searches for the best / forced-mate line
4. Switch to the **Replay** tab to step through moves manually or use **▶▶ Auto** for automatic playback

### CLI

```bash
uv run python main.py
```

Runs a built-in 3-move puzzle (two-chariot scissors mate) and prints the solution to the terminal.

## Project Structure

```
chinese-chess/
├── app.py                        # Streamlit web application
├── main.py                       # CLI entry point
├── pyproject.toml                # Project config (uv)
├── src/chinese_chess/
│   ├── pieces.py                 # Color & PieceType enums, piece symbols
│   ├── board.py                  # Board class, move generation, Zobrist hashing
│   └── solver.py                 # Alpha-beta search, evaluation, solve_timed()
└── tests/
    └── test_solver.py            # Pytest unit tests
```

## Board Encoding (JSON save format)

Positions are saved as:

```json
{
  "grid": [[0,0,0,...], ...],   // 10 rows × 9 columns
  "turn": "RED"                 // or "BLACK"
}
```

Integer encoding: `RED pieces = +1…+7`, `BLACK pieces = −1…−7`, empty = `0`.

| Value | RED | BLACK |
|-------|-----|-------|
| ±1 | 帥 King | 將 King |
| ±2 | 仕 Advisor | 士 Advisor |
| ±3 | 相 Elephant | 象 Elephant |
| ±4 | 傌 Horse | 馬 Horse |
| ±5 | 俥 Chariot | 車 Chariot |
| ±6 | 砲 Cannon | 炮 Cannon |
| ±7 | 兵 Pawn | 卒 Pawn |

Row 0 = BLACK's back rank (top), Row 9 = RED's back rank (bottom).

## Development

```bash
uv run ruff check . --fix && uv run ruff format .   # lint + format
uv run mypy src/                                     # type check
uv run pytest                                        # run tests
```

## License

MIT
