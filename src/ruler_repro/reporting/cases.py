from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..data.manifest import latest_by_key, read_jsonl, write_jsonl
from ..schemas import ExperimentConfig, ModelResponse, SampleRecord, ScoreRecord


def export_cases(configs: list[ExperimentConfig], minimum_cases: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    correct_candidates: list[dict[str, Any]] = []
    for config in configs:
        samples_path = config.paths.data_dir / "manifests" / config.model.model_id / config.experiment_id / "samples.jsonl"
        responses_path = config.paths.results_dir / "raw" / config.model.model_id / config.experiment_id / "responses.jsonl"
        scores_path = config.paths.results_dir / "scored" / config.model.model_id / config.experiment_id / "scores.jsonl"
        samples = {
            row["sample_id"]: SampleRecord.from_dict(row) for row in read_jsonl(samples_path)
        }
        response_rows = read_jsonl(responses_path) if responses_path.exists() else []
        responses = {
            key: ModelResponse.from_dict(value)
            for key, value in latest_by_key(response_rows, "sample_id").items()
        }
        scores = [ScoreRecord.from_dict(row) for row in read_jsonl(scores_path)]
        for score in scores:
            sample = samples[score.sample_id]
            response = responses.get(score.sample_id)
            case = _make_case(sample, response, score)
            if score.failure_category is not None:
                rows.append(case)
            elif score.exact_match:
                correct_candidates.append(case)

    if len(rows) < minimum_cases:
        correct_candidates.sort(
            key=lambda item: (item["source_input_tokens"], item["answer_token_position"]),
            reverse=True,
        )
        needed = minimum_cases - len(rows)
        for case in correct_candidates[:needed]:
            case["failure_category"] = "correct_boundary_case"
            case["decision_note"] = "严格匹配正确，作为较长输入或较后证据位置的边界案例。"
            rows.append(case)

    output_dir = configs[0].paths.results_dir / "cases"
    write_jsonl(output_dir / "cases.jsonl", rows, overwrite=True)
    _write_summary_csv(output_dir / "cases_summary.csv", rows)
    for model_id in sorted({config.model.model_id for config in configs}):
        model_rows = [row for row in rows if row["model_id"] == model_id]
        model_dir = output_dir / model_id
        write_jsonl(model_dir / "cases.jsonl", model_rows, overwrite=True)
        _write_summary_csv(model_dir / "cases_summary.csv", model_rows)
    return rows


def _make_case(
    sample: SampleRecord,
    response: ModelResponse | None,
    score: ScoreRecord,
) -> dict[str, Any]:
    return {
        "case_id": f"case-{sample.sample_id}",
        "model_id": sample.model_id,
        "sample_id": sample.sample_id,
        "experiment_id": sample.experiment_id,
        "task_name": sample.task_name,
        "target_length": sample.target_length,
        "num_needle_k": sample.num_needle_k,
        "source_input_tokens": sample.source_input_tokens,
        "answer_token_position": sample.answer_token_position,
        "target_key": sample.target_key,
        "reference_answers": sample.reference_answers,
        "model_status": response.status if response else "missing_response",
        "raw_prediction": response.raw_text if response else None,
        "exact_match": score.exact_match,
        "ruler_match": score.ruler_match,
        "failure_category": score.failure_category,
        "decision_note": score.decision_note,
        "input": sample.input,
        "answer_prefix": sample.answer_prefix,
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id", "model_id", "sample_id", "experiment_id", "target_length", "num_needle_k",
        "source_input_tokens", "answer_token_position", "target_key",
        "reference_answers", "model_status", "raw_prediction", "exact_match",
        "ruler_match", "failure_category", "decision_note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    temporary.replace(path)
