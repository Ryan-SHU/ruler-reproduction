from __future__ import annotations

import csv
import math
import statistics
from collections import Counter
from pathlib import Path

from ..data.manifest import latest_by_key, read_jsonl
from ..schemas import ExperimentConfig, ExperimentSummary, ModelResponse, SampleRecord, ScoreRecord


def summarize_experiment(config: ExperimentConfig) -> ExperimentSummary:
    samples_path = config.paths.data_dir / "manifests" / config.model.model_id / config.experiment_id / "samples.jsonl"
    responses_path = config.paths.results_dir / "raw" / config.model.model_id / config.experiment_id / "responses.jsonl"
    scores_path = config.paths.results_dir / "scored" / config.model.model_id / config.experiment_id / "scores.jsonl"
    samples = [SampleRecord.from_dict(row) for row in read_jsonl(samples_path)]
    response_rows = read_jsonl(responses_path) if responses_path.exists() else []
    responses = [
        ModelResponse.from_dict(row)
        for row in latest_by_key(response_rows, "sample_id").values()
    ]
    all_responses = [ModelResponse.from_dict(row) for row in response_rows]
    score_rows = read_jsonl(scores_path) if scores_path.exists() else []
    scores = [ScoreRecord.from_dict(row) for row in score_rows]
    valid = [score for score in scores if score.valid_for_metric]
    lengths = [sample.source_input_tokens for sample in samples]
    api_inputs = [item.input_tokens for item in all_responses if item.input_tokens is not None]
    api_outputs = [item.output_tokens for item in all_responses if item.output_tokens is not None]
    exact_correct = sum(score.exact_match for score in valid)
    ruler_correct = sum(score.ruler_match for score in valid)
    valid_count = len(valid)
    status_counts = Counter(response.status for response in responses)
    total_attempts = sum(response.attempt_count for response in all_responses)
    return ExperimentSummary(
        model_id=config.model.model_id,
        model_name=config.model.model_name,
        experiment_id=config.experiment_id,
        task_name=config.task_name,
        target_length=config.max_seq_length,
        num_needle_k=config.num_needle_k,
        planned_samples=config.num_samples,
        response_records=len(responses),
        successful_calls=status_counts.get("success", 0),
        valid_scored_samples=valid_count,
        exact_correct=exact_correct,
        exact_match_rate=round(exact_correct / valid_count * 100, 2) if valid_count else None,
        exact_ci95_low=_wilson_ci(exact_correct, valid_count)[0],
        exact_ci95_high=_wilson_ci(exact_correct, valid_count)[1],
        ruler_correct=ruler_correct,
        ruler_match_rate=round(ruler_correct / valid_count * 100, 2) if valid_count else None,
        ruler_ci95_low=_wilson_ci(ruler_correct, valid_count)[0],
        ruler_ci95_high=_wilson_ci(ruler_correct, valid_count)[1],
        total_attempts=total_attempts,
        retry_count=max(0, total_attempts - len(responses)),
        api_input_tokens=sum(api_inputs) if api_inputs else None,
        api_output_tokens=sum(api_outputs) if api_outputs else None,
        source_input_min=min(lengths) if lengths else None,
        source_input_median=statistics.median(lengths) if lengths else None,
        source_input_max=max(lengths) if lengths else None,
        source_input_mean=round(statistics.mean(lengths), 2) if lengths else None,
        status_counts=dict(status_counts),
    )


def write_summaries(configs: list[ExperimentConfig]) -> list[ExperimentSummary]:
    summaries = [summarize_experiment(config) for config in configs]
    for config, summary in zip(configs, summaries):
        if not config.formal:
            continue
        scores_path = (
            config.paths.results_dir / "scored" / config.model.model_id
            / config.experiment_id / "scores.jsonl"
        )
        score_count = len(read_jsonl(scores_path)) if scores_path.exists() else 0
        if summary.response_records != config.num_samples or score_count != config.num_samples:
            raise RuntimeError(
                f"{config.model.model_id}/{config.experiment_id} 尚不完整："
                f"计划 {config.num_samples}，响应 {summary.response_records}，评分 {score_count}。"
            )
    results_dir = configs[0].paths.results_dir
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = [summary.to_dict() for summary in summaries]
    is_ablation = all(config.ablation_factor is not None for config in configs)
    metrics_name = "position_ablation_metrics.csv" if is_ablation else "model_experiment_metrics.csv"
    plot_name = "position_ablation_plot_data.csv" if is_ablation else "plot_data.csv"
    _write_csv(summary_dir / metrics_name, rows)
    plot_fields = [
        "model_id", "model_name", "experiment_id", "task_name", "target_length", "num_needle_k",
        "valid_scored_samples", "exact_match_rate", "ruler_match_rate",
        "source_input_mean",
    ]
    _write_csv(
        summary_dir / plot_name,
        [{key: row[key] for key in plot_fields} for row in rows],
    )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("没有可写入的汇总记录。")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    """返回二项正确率的 Wilson 95% 置信区间，单位为百分比。"""
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return round(max(0.0, centre - margin) * 100, 2), round(min(1.0, centre + margin) * 100, 2)
