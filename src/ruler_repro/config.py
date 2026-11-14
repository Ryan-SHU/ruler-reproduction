from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .schemas import (
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
    PathsConfig,
    RetryConfig,
)


class ConfigError(ValueError):
    """实验配置无法安全执行。"""


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            missing.append(name)
            return match.group(0)
        return os.environ[name]

    expanded = _ENV_PATTERN.sub(replace, value)
    if missing:
        raise ConfigError(f"缺少环境变量：{', '.join(sorted(set(missing)))}")
    return expanded


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"配置顶层必须是映射：{path}")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _find_project_root(config_path: Path) -> Path:
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "ruler_repro").is_dir():
            return candidate.resolve()
    raise ConfigError("无法从配置路径定位 paper_reproduction/code 项目根目录。")


def load_experiment_config(
    path: str | Path,
    model_config_path: str | Path | None = None,
) -> ExperimentConfig:
    config_path = Path(path).resolve()
    experiment = _read_yaml(config_path)

    base_ref = experiment.pop("base_config", None)
    model_ref = experiment.pop("model_config", None)
    if not base_ref or not model_ref:
        raise ConfigError("实验配置必须声明 base_config 和 model_config。")

    base = _read_yaml((config_path.parent / str(base_ref)).resolve())
    selected_model_path = (
        Path(model_config_path).resolve()
        if model_config_path is not None
        else (config_path.parent / str(model_ref)).resolve()
    )
    model = _read_yaml(selected_model_path)
    merged = _expand_environment(_deep_merge(base, experiment))
    merged["model"] = _expand_environment(model)

    required = [
        "experiment_id", "task_name", "max_seq_length",
        "num_samples", "random_seed", "tokenizer_type",
    ]
    missing = [key for key in required if key not in merged]
    if missing:
        raise ConfigError(f"缺少配置字段：{', '.join(missing)}")

    project_root = _find_project_root(config_path)
    paths_value = merged.get("paths", {})
    paths = PathsConfig(
        project_root=project_root,
        upstream_dir=(project_root / paths_value.get("upstream_dir", "ref/RULER")).resolve(),
        data_dir=(project_root / paths_value.get("data_dir", "data")).resolve(),
        results_dir=(project_root / paths_value.get("results_dir", "results")).resolve(),
    )
    model_value = merged["model"]
    generation_value = merged.get("generation", {})
    retry_value = merged.get("retry", {})
    config = ExperimentConfig(
        experiment_id=str(merged["experiment_id"]),
        task_name=str(merged["task_name"]),
        max_seq_length=int(merged["max_seq_length"]),
        num_needle_k=(int(merged["num_needle_k"]) if merged.get("num_needle_k") is not None else None),
        num_chains=(int(merged["num_chains"]) if merged.get("num_chains") is not None else None),
        num_hops=(int(merged["num_hops"]) if merged.get("num_hops") is not None else None),
        num_samples=int(merged["num_samples"]),
        random_seed=int(merged["random_seed"]),
        tokenizer_type=str(merged["tokenizer_type"]),
        formal=bool(merged.get("formal", True)),
        base_experiment_id=merged.get("base_experiment_id"),
        ablation_factor=merged.get("ablation_factor"),
        evidence_position_percent=(
            int(merged["evidence_position_percent"])
            if merged.get("evidence_position_percent") is not None else None
        ),
        model=ModelConfig(
            model_id=str(model_value["model_id"]),
            model_name=str(model_value["model_name"]),
            base_url=str(model_value["base_url"]).rstrip("/"),
            tokenizer_path=str(model_value["tokenizer_path"]),
            prompt_template_type=str(model_value.get("prompt_template_type", "base")),
            context_window=int(model_value.get("context_window", 32768)),
            api_key_env=model_value.get("api_key_env"),
            request_timeout_seconds=float(model_value.get("request_timeout_seconds", 180)),
        ),
        generation=GenerationConfig(**generation_value),
        retry=RetryConfig(**retry_value),
        paths=paths,
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.task_name not in {"niah_single_2", "niah_multikey_1", "vt"}:
        raise ConfigError(f"当前范围不支持任务：{config.task_name}")
    if config.task_name == "vt":
        if config.max_seq_length not in {8192, 16384, 32768}:
            raise ConfigError("变量追踪的 max_seq_length 必须是 8192、16384 或 32768。")
        if config.num_needle_k is not None:
            raise ConfigError("变量追踪配置不使用 num_needle_k。")
        if config.num_chains != 1 or config.num_hops != 4:
            raise ConfigError("变量追踪固定使用作者设置 num_chains=1、num_hops=4。")
    else:
        if config.max_seq_length not in {2048, 4096, 8192}:
            raise ConfigError("NIAH 的 max_seq_length 必须是 2048、4096 或 8192。")
        if config.num_needle_k not in {1, 4, 8}:
            raise ConfigError("NIAH 的 num_needle_k 必须是 1、4 或 8。")
        if config.num_chains is not None or config.num_hops is not None:
            raise ConfigError("NIAH 配置不使用 num_chains 或 num_hops。")
    if config.num_samples <= 0:
        raise ConfigError("num_samples 必须大于 0。")
    if config.tokenizer_type != "hf":
        raise ConfigError("本实验固定使用 Hugging Face 分词器类型 hf。")
    expected_output_tokens = 30 if config.task_name == "vt" else 128
    if config.generation.max_output_tokens != expected_output_tokens:
        raise ConfigError(
            f"{config.task_name} 必须沿用作者的 {expected_output_tokens} 个预留输出词元。"
        )
    if config.model.context_window < config.max_seq_length:
        raise ConfigError("模型标称上下文小于实验总序列预算。")
    if config.generation.temperature != 0:
        raise ConfigError("正式复现固定使用 temperature=0。")
    if config.retry.max_attempts < 1:
        raise ConfigError("retry.max_attempts 必须至少为 1。")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config.experiment_id):
        raise ConfigError("experiment_id 只能包含字母、数字、点、下划线和连字符。")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config.model.model_id):
        raise ConfigError("model_id 只能包含字母、数字、点、下划线和连字符。")
    if config.model.prompt_template_type not in {"base", "meta-llama3"}:
        raise ConfigError("当前只支持 base 与 meta-llama3 输入模板。")
    if config.ablation_factor is not None:
        if config.ablation_factor != "evidence_position":
            raise ConfigError("当前只支持 evidence_position 消融。")
        if config.task_name != "niah_single_2" or config.num_needle_k != 1:
            raise ConfigError("证据位置消融必须使用单针任务。")
        if config.max_seq_length != 8192:
            raise ConfigError("证据位置消融固定使用 8192 总序列预算。")
        if config.evidence_position_percent not in {10, 50, 90}:
            raise ConfigError("证据位置必须是 10、50 或 90。")
        if not config.base_experiment_id:
            raise ConfigError("消融配置必须声明 base_experiment_id。")


def discover_configs(config_dir: str | Path, include_pilot: bool = False) -> list[Path]:
    paths = sorted(Path(config_dir).glob("*.yaml"))
    selected: list[Path] = []
    for path in paths:
        raw = _read_yaml(path)
        if include_pilot or bool(raw.get("formal", True)):
            selected.append(path)
    if not selected:
        raise ConfigError(f"目录中没有可用实验配置：{config_dir}")
    selected.sort(
        key=lambda path: (
            int(_read_yaml(path).get("max_seq_length", 0)),
            path.name,
        )
    )
    return selected
