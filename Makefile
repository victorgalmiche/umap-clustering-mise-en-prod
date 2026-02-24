
# ========================
# Installation / Dev Setup
# ========================
install:
	uv sync --group dev

install-prod:
	uv sync

# ========================
# Tests
# ========================
test:
	uv run pytest tests 
coverage:
	uv run pytest --cov=src/umapverse tests

# ========================
# Lint / Formatting
# ========================
lint:
	uv run ruff check src tests

# ========================
# Bump version
# ========================
bump-patch:
	uv version patch --tag

bump-minor:
	uv version minor --tag

bump-major:
	uv version major --tag