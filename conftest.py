"""Repo-root conftest: its presence puts the repo root on sys.path, so tests import `app` directly."""

import os

os.environ.setdefault("WARM_MODELS", "0")  # never load real models in the background during tests

# Third-party code and weights are not our tests.
collect_ignore = ["vendor", "models", "design", "web", ".venv", ".uv", "data"]


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads real models or calls live services (set RUN_SLOW=1)")
