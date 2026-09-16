# Alex Quant Lab

- Research only: no orders, paper trading, shorts or leverage.
- Keep credentials in environment variables or ignored `.env` files.
- Never replace missing real market data with synthetic data. Fixture mode is explicit.
- Features use information available by the signal close; trade no earlier than next open.
- Training labels must end strictly before the model's prediction block begins.
- Preserve chronological splits, complete session grids and deterministic seeds.
- Do not forward-fill missing bars or discard missing symbols silently.
- Record data hashes, configuration, model training dates and execution assumptions.
- Run `uv run --frozen pytest` and `uv run --frozen ruff check .` after changes.
- Keep generated artifacts out of Git. Commit `uv.lock` for reproducible dependencies.
- No commits, pushes, pull requests or deployments without explicit user instruction.

