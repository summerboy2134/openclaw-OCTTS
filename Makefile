PYTHON := .venv/bin/python
PIP := .venv/bin/pip
UVICORN := .venv/bin/uvicorn
PYTEST := .venv/bin/pytest
BOOTSTRAP_STAMP := .venv/.octts_bootstrapped

.PHONY: bootstrap dev run test docker-up docker-down

bootstrap:
	@test -f .env || cp .env.example .env
	@test -d .venv || python3 -m venv .venv
	@test -f $(BOOTSTRAP_STAMP) || ( $(PIP) install --upgrade pip && $(PIP) install -e '.[dev]' && touch $(BOOTSTRAP_STAMP) )

dev: bootstrap
	$(UVICORN) octts.api:app --host 0.0.0.0 --port 8000 --reload

run: bootstrap
	$(UVICORN) octts.api:app --host 0.0.0.0 --port 8000

test: bootstrap
	$(PYTEST)

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
