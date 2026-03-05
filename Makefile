# =============================================================================
# SmartTalker — Makefile
# =============================================================================

.PHONY: setup build run dev test clean logs download-models lint format help

# Default target
.DEFAULT_GOAL := help

# Variables
DOCKER_COMPOSE := docker compose
PYTHON := python3
PIP := pip
APP_NAME := smarttalker

# ---------------------------------------------------------------------------
# Setup & Build
# ---------------------------------------------------------------------------

setup: ## Initial setup: create venv, install deps, create dirs
	@echo "🚀 Setting up SmartTalker..."
	$(PYTHON) -m venv venv
	. venv/bin/activate && $(PIP) install --upgrade pip setuptools wheel
	. venv/bin/activate && $(PIP) install -r requirements.txt
	@mkdir -p models avatars voices outputs logs files
	@echo "✅ Setup complete. Activate venv with: source venv/bin/activate"

setup-win: ## Setup for Windows (PowerShell)
	@echo "🚀 Setting up SmartTalker (Windows)..."
	$(PYTHON) -m venv venv
	.\venv\Scripts\pip install --upgrade pip setuptools wheel
	.\venv\Scripts\pip install -r requirements.txt
	@if not exist models mkdir models
	@if not exist avatars mkdir avatars
	@if not exist voices mkdir voices
	@if not exist outputs mkdir outputs
	@if not exist logs mkdir logs
	@if not exist files mkdir files
	@echo "✅ Setup complete. Activate venv with: .\venv\Scripts\activate"

build: ## Build Docker images
	@echo "🔨 Building Docker images..."
	$(DOCKER_COMPOSE) build

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run: ## Run all services with Docker Compose
	@echo "🚀 Starting SmartTalker stack..."
	$(DOCKER_COMPOSE) up -d
	@echo "✅ Services running. API at http://localhost:8000"
	@echo "📖 Docs at http://localhost:8000/docs"

dev: ## Run locally in development mode (no Docker)
	@echo "🔧 Starting development server..."
	. venv/bin/activate && uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

stop: ## Stop all Docker services
	@echo "🛑 Stopping SmartTalker..."
	$(DOCKER_COMPOSE) down

# ---------------------------------------------------------------------------
# Test & Quality
# ---------------------------------------------------------------------------

test: ## Run test suite
	@echo "🧪 Running tests..."
	. venv/bin/activate && pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	. venv/bin/activate && pytest tests/ -v --cov=src --cov-report=term-missing

lint: ## Run linters (ruff + mypy)
	@echo "🔍 Linting..."
	. venv/bin/activate && ruff check src/ tests/
	. venv/bin/activate && mypy src/ --ignore-missing-imports

format: ## Format code (ruff)
	@echo "🎨 Formatting..."
	. venv/bin/activate && ruff format src/ tests/

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

download-models: ## Download all AI models
	@echo "📦 Downloading models..."
	bash scripts/download_models.sh

logs: ## Tail Docker logs
	$(DOCKER_COMPOSE) logs -f --tail=100

clean: ## Remove generated files, caches, and containers
	@echo "🧹 Cleaning up..."
	$(DOCKER_COMPOSE) down -v --remove-orphans 2>/dev/null || true
	rm -rf outputs/* logs/* files/*
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Clean complete"

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help: ## Show this help message
	@echo ""
	@echo "  SmartTalker — Digital Human AI Agent"
	@echo "  ======================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
