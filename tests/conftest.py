from __future__ import annotations

from pathlib import Path

import pytest

from memory_poisoning_sanity.models import ExperimentConfig


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def project_config(project_root: Path) -> ExperimentConfig:
    return ExperimentConfig.load(project_root / "config.yaml")
