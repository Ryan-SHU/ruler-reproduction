from __future__ import annotations

import re

from ..schemas import ExperimentConfig, SampleRecord


class SampleValidationError(ValueError):
    """生成样本不满足正式实验约束。"""


def validate_samples(samples: list[SampleRecord], config: ExperimentConfig) -> None:
    if len(samples) != config.num_samples:
        raise SampleValidationError(
            f"样本数为 {len(samples)}，配置要求 {config.num_samples}。"
        )
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise SampleValidationError("样本清单中存在重复 sample_id。")
    inputs = [sample.input for sample in samples]
    if len(inputs) != len(set(inputs)):
        raise SampleValidationError("样本清单中存在完全重复的输入。")

    for sample in samples:
        if sample.model_id not in {config.model.model_id, "legacy"}:
            raise SampleValidationError(f"{sample.sample_id} 的模型编号与配置不一致。")
        if len(sample.reference_answers) != 1:
            raise SampleValidationError(f"{sample.sample_id} 的标准答案数量不是 1。")
        answer = sample.reference_answers[0]
        if not re.fullmatch(r"\d{7}", answer):
            raise SampleValidationError(f"{sample.sample_id} 的标准答案不是七位数字。")
        if answer not in sample.input:
            raise SampleValidationError(f"{sample.sample_id} 的标准答案未出现在输入中。")
        if sample.source_total_tokens > config.max_seq_length:
            raise SampleValidationError(f"{sample.sample_id} 超过目标总序列预算。")
        expected_total = sample.source_input_tokens + config.generation.max_output_tokens
        if expected_total > config.max_seq_length:
            raise SampleValidationError(f"{sample.sample_id} 的复算总长度超过预算。")
        if sample.source_index < 0 or sample.answer_token_position < 0:
            raise SampleValidationError(f"{sample.sample_id} 的答案位置无效。")
        if config.ablation_factor == "evidence_position":
            if sample.base_sample_id is None:
                raise SampleValidationError(f"{sample.sample_id} 缺少基础样本编号。")
            if sample.evidence_position_target_percent != config.evidence_position_percent:
                raise SampleValidationError(f"{sample.sample_id} 的目标位置与配置不一致。")
            if sample.evidence_position_actual_percent is None:
                raise SampleValidationError(f"{sample.sample_id} 缺少实际证据位置。")
