$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
uv run --frozen ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv run --frozen ruff format --check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv run --frozen pytest
exit $LASTEXITCODE
