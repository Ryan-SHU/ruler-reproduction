from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    model_name: str
    base_url: str
    tokenizer_path: str
    prompt_template_type: str = "base"
    context_window: int = 32768
    api_key_env: str | None = None
    request_timeout_seconds: float = 180.0


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 128
    seed: int = 42


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    initial_wait_seconds: float = 2.0
    max_wait_seconds: float = 15.0


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path
    upstream_dir: Path
    data_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    task_name: str
    max_seq_length: int
    num_needle_k: int
    num_samples: int
    random_seed: int
    tokenizer_type: str
    model: ModelConfig
    generation: GenerationConfig
    retry: RetryConfig
    paths: PathsConfig
    formal: bool = True
    base_experiment_id: str | None = None
    ablation_factor: str | None = None
    evidence_position_percent: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["paths"] = {key: str(path) for key, path in value["paths"].items()}
        return value


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    experiment_id: str
    task_name: str
    target_length: int
    num_needle_k: int
    input: str
    answer_prefix: str
    reference_answers: list[str]
    target_key: str
    source_input_tokens: int
    source_total_tokens: int
    answer_token_position: int
    random_seed: int
    source_index: int
    model_id: str = "legacy"
    base_sample_id: str | None = None
    ablation_factor: str | None = None
    evidence_position_target_percent: int | None = None
    evidence_position_actual_percent: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SampleRecord":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelRequest:
    sample_id: str
    model_name: str
    prompt: str
    temperature: float
    top_p: float
    max_output_tokens: int
    seed: int


@dataclass(frozen=True)
class ModelResponse:
    sample_id: str
    experiment_id: str
    model_name: str
    raw_text: str | None
    status: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    attempt_count: int
    error_type: str | None = None
    error_message: str | None = None
    model_id: str = "legacy"
    prompt_template_type: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelResponse":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreRecord:
    sample_id: str
    experiment_id: str
    normalized_prediction: str
    exact_match: bool
    ruler_match: bool
    valid_for_metric: bool
    failure_category: str | None
    decision_note: str
    model_id: str = "legacy"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScoreRecord":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentSummary:
    model_id: str
    model_name: str
    experiment_id: str
    task_name: str
    target_length: int
    num_needle_k: int
    planned_samples: int
    response_records: int
    successful_calls: int
    valid_scored_samples: int
    exact_correct: int
    exact_match_rate: float | None
    exact_ci95_low: float | None
    exact_ci95_high: float | None
    ruler_correct: int
    ruler_match_rate: float | None
    ruler_ci95_low: float | None
    ruler_ci95_high: float | None
    total_attempts: int
    retry_count: int
    api_input_tokens: int | None
    api_output_tokens: int | None
    source_input_min: int | None
    source_input_median: float | None
    source_input_max: int | None
    source_input_mean: float | None
    status_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["status_counts"] = ";".join(
            f"{key}:{value}" for key, value in sorted(self.status_counts.items())
        )
        return row
