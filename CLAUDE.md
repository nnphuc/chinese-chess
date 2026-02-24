# Chinese Chess Project Rules

## Code Quality

Always run these before finishing any code change:

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy src/             # type check
uv run pytest                # tests
```

Fix all ruff and mypy errors before considering a task done.

## Tooling

- **Linter/formatter**: ruff (`uv run ruff check . --fix && uv run ruff format .`)
- **Type checker**: mypy (`uv run mypy src/`)
- **Tests**: pytest (`uv run pytest`)

## Style

- Follow ruff defaults (PEP 8, isort)
- All public functions must have type annotations
- Tests go in `tests/` directory, named `test_*.py`
