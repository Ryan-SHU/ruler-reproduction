from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from ..schemas import ExperimentConfig, SampleRecord
from .manifest import read_jsonl, write_jsonl
from .validator import validate_samples


class PositionAblationError(RuntimeError):
    """证据位置消融样本无法安全构造。"""


_NEEDLE_PATTERN = re.compile(
    r"One of the special magic numbers for .+? is: \d{7}\."
)
_CONTEXT_START = "afterwards.\n"
_QUESTION_START = "\nWhat "


def _sent_tokenize(text: str) -> list[str]:
    from nltk.tokenize import sent_tokenize

    return sent_tokenize(text)


def _manifest_path(config: ExperimentConfig, experiment_id: str) -> Path:
    return (
        config.paths.data_dir / "manifests" / config.model.model_id
        / experiment_id / "samples.jsonl"
    )


def _relocate(sample: SampleRecord, config: ExperimentConfig, tokenizer: object) -> SampleRecord:
    matches = list(_NEEDLE_PATTERN.finditer(sample.input))
    if len(matches) != 1:
        raise PositionAblationError(
            f"{sample.sample_id} 应包含一条证据句，实际为 {len(matches)} 条。"
        )
    start_marker = sample.input.find(_CONTEXT_START)
    question_marker = sample.input.rfind(_QUESTION_START)
    if start_marker < 0 or question_marker < 0 or start_marker >= question_marker:
        raise PositionAblationError(f"{sample.sample_id} 无法定位上下文边界。")
    context_start = start_marker + len(_CONTEXT_START)
    context = sample.input[context_start:question_marker]
    needle_match = _NEEDLE_PATTERN.search(context)
    if needle_match is None:
        raise PositionAblationError(f"{sample.sample_id} 的证据句不在上下文区域。")
    needle = needle_match.group(0)
    background = (context[:needle_match.start()] + context[needle_match.end():]).strip()
    sentences = _sent_tokenize(background)
    if not sentences:
        raise PositionAblationError(f"{sample.sample_id} 的背景文本为空。")
    target = config.evidence_position_percent
    if target is None:
        raise PositionAblationError("消融配置缺少目标位置。")
    insertion_index = round(len(sentences) * target / 100)
    relocated_sentences = sentences[:insertion_index] + [needle] + sentences[insertion_index:]
    relocated_context = " ".join(relocated_sentences)
    relocated_input = (
        sample.input[:context_start] + relocated_context + sample.input[question_marker:]
    )
    prompt = relocated_input + sample.answer_prefix
    source_input_tokens = len(tokenizer.tokenize(prompt))
    answer_character_position = relocated_input.find(sample.reference_answers[0])
    answer_token_position = len(tokenizer.tokenize(relocated_input[:answer_character_position]))
    actual_percent = round(answer_token_position / max(source_input_tokens, 1) * 100, 2)
    return replace(
        sample,
        sample_id=(
            f"{config.model.model_id}-{config.experiment_id}-"
            f"{config.random_seed}-{sample.sample_id.rsplit('-', 1)[-1]}"
        ),
        experiment_id=config.experiment_id,
        input=relocated_input,
        source_input_tokens=source_input_tokens,
        source_total_tokens=source_input_tokens + config.generation.max_output_tokens,
        answer_token_position=answer_token_position,
        base_sample_id=sample.sample_id,
        ablation_factor="evidence_position",
        evidence_position_target_percent=target,
        evidence_position_actual_percent=actual_percent,
    )


def generate_position_ablation(
    config: ExperimentConfig,
    force: bool = False,
) -> list[SampleRecord]:
    if not config.base_experiment_id:
        raise PositionAblationError("消融配置缺少 base_experiment_id。")
    base_path = _manifest_path(config, config.base_experiment_id)
    output_path = _manifest_path(config, config.experiment_id)
    if output_path.exists() and not force:
        samples = [SampleRecord.from_dict(row) for row in read_jsonl(output_path)]
        validate_samples(samples, config)
        return samples
    if not base_path.is_file():
        raise FileNotFoundError(
            f"缺少同模型的基础样本：{base_path}。请先 prepare single_8k。"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_path,
        trust_remote_code=True,
    )
    base_samples = [SampleRecord.from_dict(row) for row in read_jsonl(base_path)]
    samples = [_relocate(sample, config, tokenizer) for sample in base_samples]
    validate_samples(samples, config)
    write_jsonl(output_path, samples, overwrite=force)
    return samples
