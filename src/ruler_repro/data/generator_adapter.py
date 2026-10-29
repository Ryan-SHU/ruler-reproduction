from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from ..schemas import ExperimentConfig, SampleRecord
from ..inference.prompt_renderer import render_author_prompt_template
from .manifest import read_jsonl, write_jsonl
from .validator import validate_samples


class GeneratorError(RuntimeError):
    """作者数据生成器未正常完成。"""


def _load_python_mapping(path: Path, variable: str) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(f"ruler_source_{variable}", path)
    if spec is None or spec.loader is None:
        raise GeneratorError(f"无法读取上游 Python 配置：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = getattr(module, variable, None)
    if not isinstance(value, dict):
        raise GeneratorError(f"上游文件缺少字典 {variable}：{path}")
    return value


def _extract_target_key(answer_prefix: str) -> str:
    match = re.search(r"\bfor (.+?) mentioned in the provided text", answer_prefix)
    if not match:
        raise GeneratorError("无法从作者回答前缀解析目标键。")
    return match.group(1).strip()


def _source_paths(config: ExperimentConfig) -> tuple[Path, Path, Path]:
    generated_dir = config.paths.data_dir / "generated" / config.model.model_id / config.experiment_id
    source_path = generated_dir / "source.jsonl"
    manifest_path = (
        config.paths.data_dir / "manifests" / config.model.model_id
        / config.experiment_id / "samples.jsonl"
    )
    author_path = generated_dir / "_author" / "validation.jsonl"
    return source_path, manifest_path, author_path


def generate_samples(config: ExperimentConfig, force: bool = False) -> list[SampleRecord]:
    source_path, manifest_path, author_path = _source_paths(config)
    if manifest_path.exists() and not force:
        samples = [SampleRecord.from_dict(row) for row in read_jsonl(manifest_path)]
        validate_samples(samples, config)
        return samples

    scripts_dir = config.paths.upstream_dir / "scripts"
    niah_path = scripts_dir / "data" / "synthetic" / "niah.py"
    synthetic_yaml = scripts_dir / "synthetic.yaml"
    constants_path = scripts_dir / "data" / "synthetic" / "constants.py"
    templates_path = scripts_dir / "data" / "template.py"
    background_path = scripts_dir / "data" / "synthetic" / "json" / "PaulGrahamEssays.json"
    for required in [niah_path, synthetic_yaml, constants_path, templates_path, background_path]:
        if not required.is_file():
            raise GeneratorError(f"缺少上游生成器所需文件：{required}")

    with synthetic_yaml.open("r", encoding="utf-8") as handle:
        task_configs = yaml.safe_load(handle)
    if config.task_name not in task_configs:
        raise GeneratorError(f"上游 synthetic.yaml 中没有任务 {config.task_name}。")
    task_config = task_configs[config.task_name]
    if task_config.get("task") != "niah":
        raise GeneratorError("当前适配器只允许 NIAH 任务。")
    task_args = dict(task_config.get("args", {}))
    task_args["num_needle_k"] = config.num_needle_k

    tasks = _load_python_mapping(constants_path, "TASKS")
    templates = _load_python_mapping(templates_path, "Templates")
    niah = tasks["niah"]
    template_type = config.model.prompt_template_type
    prompt_template = render_author_prompt_template(
        templates,
        template_type,
        niah["template"],
        niah["answer_prefix"],
    )

    if author_path.exists() and not force:
        raw_rows = read_jsonl(author_path)
    else:
        command = [
            sys.executable,
            str(niah_path),
            "--save_dir", str(author_path.parent.parent),
            "--save_name", author_path.parent.name,
            "--subset", "validation",
            "--tokenizer_path", config.model.tokenizer_path,
            "--tokenizer_type", config.tokenizer_type,
            "--max_seq_length", str(config.max_seq_length),
            "--tokens_to_generate", str(config.generation.max_output_tokens),
            "--num_samples", str(config.num_samples),
            "--random_seed", str(config.random_seed),
            "--template", prompt_template,
        ]
        for key in [
            "num_needle_k", "num_needle_v", "num_needle_q", "type_haystack",
            "type_needle_k", "type_needle_v",
        ]:
            command.extend([f"--{key}", str(task_args[key])])
        try:
            completed = subprocess.run(
                command,
                cwd=niah_path.parent,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise GeneratorError(f"无法启动作者生成器：{exc}") from exc
        if completed.returncode != 0:
            stderr = completed.stderr[-4000:].strip()
            raise GeneratorError(f"作者生成器退出码为 {completed.returncode}：{stderr}")
        raw_rows = read_jsonl(author_path)

    if len(raw_rows) != config.num_samples:
        raise GeneratorError("作者生成器输出的样本数与配置不一致。")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path, trust_remote_code=True)
    samples: list[SampleRecord] = []
    for row_number, row in enumerate(raw_rows):
        prompt = str(row["input"]) + str(row.get("answer_prefix", ""))
        source_input_tokens = len(tokenizer.tokenize(prompt))
        sample = SampleRecord(
            sample_id=(
                f"{config.model.model_id}-{config.experiment_id}-"
                f"{config.random_seed}-{row_number:04d}"
            ),
            experiment_id=config.experiment_id,
            task_name=config.task_name,
            target_length=config.max_seq_length,
            num_needle_k=config.num_needle_k,
            input=str(row["input"]),
            answer_prefix=str(row.get("answer_prefix", "")),
            reference_answers=[str(item) for item in row["outputs"]],
            target_key=_extract_target_key(str(row.get("answer_prefix", ""))),
            source_input_tokens=source_input_tokens,
            source_total_tokens=int(row["length"]),
            answer_token_position=int(row["token_position_answer"]),
            random_seed=config.random_seed,
            source_index=int(row["index"]),
            model_id=config.model.model_id,
        )
        samples.append(sample)
    validate_samples(samples, config)
    write_jsonl(source_path, raw_rows, overwrite=force)
    write_jsonl(manifest_path, samples, overwrite=force)
    return samples
