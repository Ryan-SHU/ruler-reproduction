from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from ..data.manifest import read_jsonl
from ..schemas import ExperimentConfig, SampleRecord, ScoreRecord


def analyze_position_ablation(configs: list[ExperimentConfig]) -> tuple[Path, Path]:
    from statsmodels.stats.contingency_tables import cochrans_q, mcnemar

    selected = [config for config in configs if config.ablation_factor == "evidence_position"]
    if not selected:
        raise ValueError("没有选择证据位置消融配置。")
    grouped: dict[str, dict[str, dict[int, bool]]] = defaultdict(lambda: defaultdict(dict))
    detail_rows: list[dict[str, object]] = []
    for config in selected:
        manifest_path = config.paths.data_dir / "manifests" / config.model.model_id / config.experiment_id / "samples.jsonl"
        scores_path = config.paths.results_dir / "scored" / config.model.model_id / config.experiment_id / "scores.jsonl"
        samples = {row["sample_id"]: SampleRecord.from_dict(row) for row in read_jsonl(manifest_path)}
        scores = [ScoreRecord.from_dict(row) for row in read_jsonl(scores_path)]
        for score in scores:
            sample = samples[score.sample_id]
            if sample.base_sample_id is None or sample.evidence_position_target_percent is None:
                raise ValueError(f"消融样本字段不完整：{sample.sample_id}")
            grouped[config.model.model_id][sample.base_sample_id][sample.evidence_position_target_percent] = score.exact_match
            detail_rows.append({
                "model_id": config.model.model_id,
                "base_sample_id": sample.base_sample_id,
                "sample_id": sample.sample_id,
                "target_position_percent": sample.evidence_position_target_percent,
                "actual_position_percent": sample.evidence_position_actual_percent,
                "exact_match": score.exact_match,
                "ruler_match": score.ruler_match,
                "failure_category": score.failure_category,
            })

    summary_rows: list[dict[str, object]] = []
    positions = [10, 50, 90]
    for model_id, samples in sorted(grouped.items()):
        complete = [values for values in samples.values() if set(values) == set(positions)]
        if not complete:
            raise ValueError(f"{model_id} 没有完整的三位置配对样本。")
        matrix = [[int(values[position]) for position in positions] for values in complete]
        q_result = cochrans_q(matrix)
        summary_rows.append({
            "model_id": model_id,
            "test": "cochrans_q",
            "comparison": "10_vs_50_vs_90",
            "paired_samples": len(matrix),
            "statistic": float(q_result.statistic),
            "p_value": float(q_result.pvalue),
            "holm_threshold": "",
        })
        pair_results: list[dict[str, object]] = []
        for left, right in [(10, 50), (10, 90), (50, 90)]:
            table = [[0, 0], [0, 0]]
            for values in complete:
                table[int(values[left])][int(values[right])] += 1
            result = mcnemar(table, exact=True)
            pair_results.append({
                "model_id": model_id,
                "test": "mcnemar_exact",
                "comparison": f"{left}_vs_{right}",
                "paired_samples": len(matrix),
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            })
        ordered = sorted(pair_results, key=lambda row: float(row["p_value"]))
        for rank, row in enumerate(ordered, start=1):
            row["holm_threshold"] = round(0.05 / (len(ordered) - rank + 1), 6)
        summary_rows.extend(pair_results)

    output_dir = selected[0].paths.results_dir / "summary"
    detail_path = output_dir / "position_ablation_paired_cases.csv"
    stats_path = output_dir / "position_ablation_statistics.csv"
    _write_csv(detail_path, detail_rows)
    _write_csv(stats_path, summary_rows)
    return detail_path, stats_path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
