from __future__ import annotations

import importlib.util
from pathlib import Path

from .model.openai_compatible import OpenAICompatibleClient
from .schemas import ExperimentConfig


def run_preflight(config: ExperimentConfig, check_server: bool = False) -> list[str]:
    checks: list[tuple[str, bool, str]] = []
    upstream_files = [
        config.paths.upstream_dir / "scripts" / "synthetic.yaml",
        config.paths.upstream_dir / "scripts" / "data" / "synthetic" / "niah.py",
        config.paths.upstream_dir / "scripts" / "data" / "synthetic" / "constants.py",
        config.paths.upstream_dir / "scripts" / "data" / "template.py",
        config.paths.upstream_dir / "scripts" / "data" / "synthetic" / "json" / "PaulGrahamEssays.json",
    ]
    for path in upstream_files:
        checks.append((f"文件 {path}", path.is_file(), "缺少上游源码或文章背景数据"))
    tokenizer_path = Path(config.model.tokenizer_path)
    checks.append(
        (
            f"本地分词器/权重路径 {tokenizer_path}",
            tokenizer_path.is_dir(),
            "模型路径环境变量必须指向已经下载的模型目录",
        )
    )
    for package in ["yaml", "numpy", "nltk", "wonderwords", "transformers", "matplotlib", "statsmodels"]:
        checks.append((f"Python 依赖 {package}", importlib.util.find_spec(package) is not None, "请安装 code 项目依赖"))
    try:
        import nltk

        nltk.data.find("tokenizers/punkt")
        punkt_ok = True
    except (ImportError, LookupError):
        punkt_ok = False
    checks.append(("NLTK punkt 句子分词数据", punkt_ok, "执行 python -m nltk.downloader punkt punkt_tab"))

    messages: list[str] = []
    failed = False
    for label, passed, hint in checks:
        messages.append(f"[{'通过' if passed else '失败'}] {label}" + ("" if passed else f"；{hint}"))
        failed = failed or not passed
    if check_server:
        try:
            OpenAICompatibleClient(config.model, config.retry).health_check()
            messages.append(f"[通过] 模型服务 {config.model.base_url}")
        except Exception as exc:
            messages.append(f"[失败] 模型服务 {config.model.base_url}；{exc}")
            failed = True
    if failed:
        raise RuntimeError("\n".join(messages))
    return messages
