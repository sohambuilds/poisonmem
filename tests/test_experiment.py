from __future__ import annotations

from memory_poisoning_sanity.data import generate_data
from memory_poisoning_sanity.experiment import ExperimentRunner
from memory_poisoning_sanity.models import ExperimentConfig


def test_stage_call_counts_and_oracle_rank(tmp_path, project_root):
    generate_data(tmp_path / "data")
    config = ExperimentConfig.load(project_root / "config.yaml")
    runner = ExperimentRunner(tmp_path, config, client=None)  # type: ignore[arg-type]
    assert len(runner.preflight_specs()) == 384
    oracle = runner.oracle_specs()
    assert len(oracle) == 384
    oracle_contexts = [spec for spec in oracle if spec.condition == "oracle"]
    assert all(len(spec.memories) == 4 for spec in oracle_contexts)
    assert all(spec.memories[1].is_poison for spec in oracle_contexts)
    assert all(sum(memory.is_poison for memory in spec.memories) == 1 for spec in oracle_contexts)

