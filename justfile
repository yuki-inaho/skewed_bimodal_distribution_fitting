set shell := ["bash", "-cu"]

env := "PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"

# Create a local uv virtual environment.
venv:
    uv venv

# Install runtime and development dependencies.
sync:
    uv sync --extra dev

# Format Python source, tests, scripts, and notebook-builder files with Ruff.
format:
    uv run ruff format .

# Run Ruff lint checks.
lint:
    uv run ruff check .

# Apply safe Ruff lint fixes.
lint-fix:
    uv run ruff check . --fix

# Run Astral ty static type checks. Unresolved third-party stubs are ignored intentionally.
typecheck:
    uv run ty check src tests scripts --ignore unresolved-import --python .venv

# Run pytest-based unit and smoke tests.
test:
    {{env}} uv run pytest -q

# Run radon complexity and maintainability checks.
radon:
    uv run radon cc src scripts -s -a
    uv run radon mi src scripts -s

# Run the fixed scenario experiment.
run:
    {{env}} uv run python scripts/run_experiment.py --output-dir outputs --sample-size 400 --seed 20260503 --max-iter 60

# Run the randomized quality search experiment.
random:
    {{env}} uv run python scripts/run_random_search.py --output-dir outputs --trials 24 --sample-size 300 --seed 20260504 --max-iter 60

# Build and execute the notebook report from existing outputs.
notebook:
    {{env}} uv run python scripts/build_report_notebook.py --execute

# Full validation path.
quality: format lint typecheck test radon run random notebook

# Remove generated quality caches. Source files and generated reports remain untouched.
clean-caches:
    rm -rf .pytest_cache .ruff_cache .ty
