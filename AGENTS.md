# AGENTS.md

## Setup
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

## Definition of Done (must pass before opening PR)
ruff check .
ruff format --check .
pytest -q

## Repo rules
- Do not commit or modify local configs (config/local.*) or secrets (.env).
- Do not touch generated folders (results/, debug_out/).
- Keep PRs small and scoped; avoid unrelated refactors.
- Avoid adding new dependencies unless necessary.