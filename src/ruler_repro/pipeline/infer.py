from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..data.manifest import append_jsonl, latest_by_key, read_jsonl
from ..model.openai_compatible import OpenAICompatibleClient
from ..schemas import ExperimentConfig, ModelRequest, ModelResponse, SampleRecord


def _paths(config: ExperimentConfig) -> tuple[Path, Path, Path, Path]:
    samples = (
        config.paths.data_dir / "manifests" / config.model.model_id
        / config.experiment_id / "samples.jsonl"
    )
    output_dir = config.paths.results_dir / "raw" / config.model.model_id / config.experiment_id
    return (
        samples,
        output_dir / "responses.jsonl",
        output_dir / "run_config.yaml",
        output_dir / "run.log",
    )


def _log_event(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")


def _freeze_run_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != serialized:
            raise RuntimeError(
                f"已有运行配置与当前配置不同：{path}。请使用新的 experiment_id。"
            )
    else:
        path.write_text(serialized, encoding="utf-8")


def run_experiment(
    config: ExperimentConfig,
    retry_failed: bool = False,
    limit: int | None = None,
) -> list[ModelResponse]:
    samples_path, responses_path, run_config_path, log_path = _paths(config)
    if not samples_path.is_file():
        raise FileNotFoundError(f"尚未冻结样本，请先执行 prepare：{samples_path}")
    _freeze_run_config(config, run_config_path)
    samples = [SampleRecord.from_dict(row) for row in read_jsonl(samples_path)]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit 必须大于 0。")
        samples = samples[:limit]

    existing_rows = read_jsonl(responses_path) if responses_path.exists() else []
    existing = latest_by_key(existing_rows, "sample_id")
    client = OpenAICompatibleClient(
        config.model,
        config.retry,
        event_logger=lambda message: _log_event(log_path, message),
    )
    _log_event(log_path, f"run_start experiment_id={config.experiment_id} retry_failed={retry_failed}")
    client.health_check()

    new_responses: list[ModelResponse] = []
    for position, sample in enumerate(samples, start=1):
        previous = existing.get(sample.sample_id)
        if previous and (previous.get("status") == "success" or not retry_failed):
            print(f"[{position}/{len(samples)}] 跳过已有记录 {sample.sample_id}")
            continue
        request = ModelRequest(
            sample_id=sample.sample_id,
            model_name=config.model.model_name,
            prompt=sample.input + sample.answer_prefix,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            max_output_tokens=config.generation.max_output_tokens,
            seed=config.generation.seed,
        )
        response = client.generate(request, config.experiment_id)
        response = ModelResponse(
            **{
                **response.to_dict(),
                "model_id": config.model.model_id,
                "prompt_template_type": config.model.prompt_template_type,
            }
        )
        append_jsonl(responses_path, response)
        _log_event(
            log_path,
            f"sample_id={sample.sample_id} status={response.status} "
            f"attempt_count={response.attempt_count} latency_ms={response.latency_ms}",
        )
        new_responses.append(response)
        print(
            f"[{position}/{len(samples)}] {sample.sample_id}: "
            f"{response.status}, attempts={response.attempt_count}"
        )
    _log_event(log_path, f"run_end new_response_records={len(new_responses)}")
    return new_responses
