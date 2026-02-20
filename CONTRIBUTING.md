# Contributing to SmartTalker

Thanks for your interest in contributing to SmartTalker! This guide will help you get started.

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/<your-username>/SmartTalker.git
cd SmartTalker
```

### 2. Set Up the Dev Environment

**Linux / macOS:**

```bash
make setup
source venv/bin/activate
```

**Windows:**

```bash
make setup-win
.\venv\Scripts\activate
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your local settings
```

## Development Workflow

### Running the App

```bash
make dev
```

The API will be available at `http://localhost:8000` with docs at `http://localhost:8000/docs`.

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run a specific test file
pytest tests/test_api.py -v
```

All 171 tests must pass before submitting a PR.

### Linting and Formatting

```bash
# Check for lint errors (ruff + mypy)
make lint

# Auto-format code
make format
```

We use **ruff** for linting and formatting, and **mypy** for type checking.

## Submitting Changes

### Branch Naming

Create a branch from `master` with a descriptive name:

```
feature/add-voice-selection
fix/whatsapp-retry-logic
docs/update-api-examples
```

### Commit Messages

Write concise commit messages that explain **why**, not just what. Follow the style used in the repo:

```
Fix webhook signature validation for empty payloads
Add voice cloning endpoint with 3-10s reference audio
Remove unused imports across pipeline modules
```

- Start with an imperative verb (Fix, Add, Update, Remove, Refactor)
- Keep the first line under 72 characters
- Add a blank line and details in the body if needed

### Pull Requests

1. Make sure all tests pass (`make test`)
2. Run the linter (`make lint`)
3. Push your branch and open a PR against `master`
4. Provide a clear description of what changed and why
5. Link any related issues

## Code Style

- **Python 3.10+** — use modern syntax (type unions with `|`, `match` statements where appropriate)
- **Type hints** — add type annotations to function signatures
- **Pydantic** — use Pydantic models for all request/response schemas and validation
- **Async** — pipeline and integration code should be async where possible
- **Imports** — keep imports clean, remove unused ones

## Project Structure

```
src/
  api/          # FastAPI routes, schemas, middleware, websockets
  pipeline/     # AI engines (ASR, LLM, TTS, video, upscale)
  integrations/ # External services (WhatsApp, storage, WebRTC)
  utils/        # Logging, exceptions, audio/video helpers
  config.py     # Pydantic Settings configuration
  main.py       # Application entry point
tests/          # Test suite (mirrors src/ structure)
```

## Reporting Issues

When opening an issue, please include:

- A clear description of the problem or feature request
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Python version, OS, and GPU info (if relevant)

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
