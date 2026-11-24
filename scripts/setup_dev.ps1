$ErrorActionPreference = "Stop"

poetry install
poetry run pre-commit install

Write-Host "Development environment is ready."
Write-Host "Run: poetry run pytest"
Write-Host "Run: poetry run pre-commit run --all-files"
