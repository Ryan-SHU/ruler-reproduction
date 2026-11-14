from __future__ import annotations

import re
from pathlib import Path

from ..data.manifest import latest_by_key, read_jsonl, write_jsonl
from ..schemas import ExperimentConfig, ModelResponse, SampleRecord, ScoreRecord
from ..scoring.exact_match import ExactMatchScorer
from ..scoring.ruler_match import RulerStringMatchScorer


def _classify_failure(
    sample: SampleRecord,
    response: ModelResponse | None,
    normalized: str,
    exact: bool,
    ruler: bool,
    ruler_score: float = 0.0,
) -> tuple[str | None, str]:
    if response is None:
        return "missing_response", "没有找到该样本的模型响应。"
    if response.status != "success":
        return "call_failure", f"模型调用状态为 {response.status}。"
    if exact:
        return None, "严格匹配通过。"
    if not normalized:
        return "empty_answer", "接口成功返回，但回答去除首尾空白后为空。"
    if ruler:
        return "strict_format_failure", "回答包含完整标准答案，但含有额外字符。"
    if sample.task_name == "vt":
        if ruler_score > 0:
            matched = round(ruler_score * len(sample.reference_answers))
            return (
                "partial_retrieval",
                f"回答命中 {matched}/{len(sample.reference_answers)} 个标准变量名。",
            )
        return "unrelated_answer", "回答没有命中任何标准变量名。"
    reference = sample.reference_answers[0]
    digit_groups = set(re.findall(r"(?<!\d)\d{7}(?!\d)", sample.input))
    predicted_groups = set(re.findall(r"(?<!\d)\d{7}(?!\d)", normalized))
    distractors = digit_groups - {reference}
    if predicted_groups & distractors:
        return "distractor_confusion", "回答包含同一输入中的其他七位数字干扰项。"
    if any(part and part in reference for part in re.findall(r"\d+", normalized)):
        return "incomplete_answer", "回答中的数字片段属于标准答案的一部分。"
    return "unrelated_answer", "回答不包含标准答案，也未匹配到可识别的数字干扰项。"


def score_experiment(config: ExperimentConfig, force: bool = False) -> list[ScoreRecord]:
    samples_path = config.paths.data_dir / "manifests" / config.model.model_id / config.experiment_id / "samples.jsonl"
    responses_path = config.paths.results_dir / "raw" / config.model.model_id / config.experiment_id / "responses.jsonl"
    scores_path = config.paths.results_dir / "scored" / config.model.model_id / config.experiment_id / "scores.jsonl"
    samples = [SampleRecord.from_dict(row) for row in read_jsonl(samples_path)]
    response_rows = read_jsonl(responses_path) if responses_path.exists() else []
    responses = {
        key: ModelResponse.from_dict(value)
        for key, value in latest_by_key(response_rows, "sample_id").items()
    }
    exact_scorer = ExactMatchScorer()
    ruler_scorer = RulerStringMatchScorer()
    scores: list[ScoreRecord] = []
    for sample in samples:
        response = responses.get(sample.sample_id)
        valid = response is not None and response.status == "success"
        prediction = response.raw_text if valid and response.raw_text is not None else ""
        normalized = exact_scorer.normalize(prediction)
        exact = exact_scorer.score(prediction, sample.reference_answers) if valid else False
        ruler_score = (
            ruler_scorer.score_fraction(prediction, sample.reference_answers)
            if valid else 0.0
        )
        ruler = ruler_score == 1.0
        category, note = _classify_failure(
            sample, response, normalized, exact, ruler, ruler_score
        )
        scores.append(
            ScoreRecord(
                sample_id=sample.sample_id,
                experiment_id=config.experiment_id,
                normalized_prediction=normalized,
                exact_match=exact,
                ruler_match=ruler,
                valid_for_metric=valid,
                failure_category=category,
                decision_note=note,
                model_id=config.model.model_id,
                ruler_score=ruler_score,
            )
        )
    write_jsonl(scores_path, scores, overwrite=force or scores_path.exists())
    return scores
