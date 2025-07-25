install:
	uv sync --dev --extra jupyter --frozen

test:
	uv run pytest tests/unit && uv run pytest tests/integration

compile:
	uv run python -m pipelines.meta.pipeline

trigger:
	uv run python -m cloudtrigger --platform=meta --enable_caching=$(caching)


setup-dev-env:
	PROJECT_ID=$$(gcloud config get-value project) && \
	(cd deployment/terraform/dev && terraform init && terraform apply --var-file vars/env.tfvars --var dev_project_id=$$PROJECT_ID --auto-approve)

lint:
	uv run codespell
	uv run ruff check . --diff
	uv run ruff format . --check --diff
	uv run mypy .
