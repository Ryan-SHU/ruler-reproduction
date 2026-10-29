from __future__ import annotations

from ..data.generator_adapter import generate_samples
from ..data.position_ablation import generate_position_ablation
from ..schemas import ExperimentConfig, SampleRecord


def prepare_experiment(config: ExperimentConfig, force: bool = False) -> list[SampleRecord]:
    if config.ablation_factor == "evidence_position":
        return generate_position_ablation(config, force=force)
    return generate_samples(config, force=force)
