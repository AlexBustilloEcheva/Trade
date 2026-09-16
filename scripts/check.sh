#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest
