from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, discover_configs, load_experiment_config
from .pipeline.evaluate import score_experiment
from .pipeline.ablation import analyze_position_ablation
from .pipeline.infer import run_experiment
from .pipeline.prepare import prepare_experiment
from .pipeline.summarize import write_summaries
from .preflight import run_preflight
from .reporting.cases import export_cases
from .reporting.curves import write_figures


def _add_selection(parser: argparse.ArgumentParser, allow_dir: bool = True) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="单个实验配置文件")
    if allow_dir:
        group.add_argument("--config-dir", type=Path, help="实验配置目录")
        parser.add_argument("--include-pilot", action="store_true", help="目录模式中包含试跑配置")
    parser.add_argument(
        "--model-config",
        type=Path,
        action="append",
        help="覆盖实验文件中的模型配置；可重复传入以汇总两个模型",
    )


def _selected_paths(args: argparse.Namespace) -> list[Path]:
    if args.config:
        return [args.config]
    return discover_configs(args.config_dir, include_pilot=getattr(args, "include_pilot", False))


def _load_selected_configs(args: argparse.Namespace):
    paths = _selected_paths(args)
    model_paths = args.model_config or [None]
    return [
        load_experiment_config(path, model_path)
        for model_path in model_paths
        for path in paths
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RULER 部分复现实验流水线")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="检查数据、依赖、模型路径与可选服务")
    _add_selection(preflight, allow_dir=False)
    preflight.add_argument("--check-server", action="store_true")

    prepare = subparsers.add_parser("prepare", help="生成、校验并冻结样本")
    _add_selection(prepare)
    prepare.add_argument("--force", action="store_true", help="明确覆盖该实验已有生成数据")

    prepare_ablation = subparsers.add_parser(
        "prepare-ablation",
        help="生成并校验证据位置消融样本",
    )
    _add_selection(prepare_ablation)
    prepare_ablation.add_argument("--force", action="store_true", help="明确覆盖已有消融数据")

    run = subparsers.add_parser("run", help="调用本地模型服务并保存原始响应")
    _add_selection(run)
    run.add_argument("--retry-failed", action="store_true", help="重新请求已有失败记录")
    run.add_argument("--limit", type=int, help="只调用配置样本清单开头的若干条")

    score = subparsers.add_parser("score", help="计算逐样本评分")
    _add_selection(score)

    summarize = subparsers.add_parser("summarize", help="输出跨实验汇总和图形")
    _add_selection(summarize)

    cases = subparsers.add_parser("cases", help="导出失败与边界案例")
    _add_selection(cases)
    cases.add_argument("--minimum-cases", type=int, default=10)

    analyze = subparsers.add_parser("analyze-ablation", help="输出双模型位置消融配对统计")
    _add_selection(analyze)

    all_steps = subparsers.add_parser("all", help="依次生成、推理、评分、汇总和导出案例")
    _add_selection(all_steps)
    all_steps.add_argument("--retry-failed", action="store_true")
    all_steps.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configs = _load_selected_configs(args)
        if args.command == "preflight":
            for message in run_preflight(configs[0], check_server=args.check_server):
                print(message)
        elif args.command == "prepare":
            for config in configs:
                samples = prepare_experiment(config, force=args.force)
                print(f"{config.experiment_id}: 已冻结 {len(samples)} 条样本。")
        elif args.command == "prepare-ablation":
            for config in configs:
                if config.ablation_factor != "evidence_position":
                    raise ConfigError("prepare-ablation 只接受证据位置消融配置。")
                samples = prepare_experiment(config, force=args.force)
                print(f"{config.model.model_id}/{config.experiment_id}: 已冻结 {len(samples)} 条消融样本。")
        elif args.command == "run":
            for config in configs:
                responses = run_experiment(config, retry_failed=args.retry_failed, limit=args.limit)
                print(f"{config.experiment_id}: 本次新增 {len(responses)} 条响应。")
        elif args.command == "score":
            for config in configs:
                scores = score_experiment(config, force=True)
                print(f"{config.experiment_id}: 已评分 {len(scores)} 条样本。")
        elif args.command == "summarize":
            summaries = write_summaries(configs)
            created = write_figures(summaries, configs[0].paths.results_dir / "figures")
            print(f"已汇总 {len(summaries)} 个实验并生成 {len(created)} 张图。")
        elif args.command == "cases":
            rows = export_cases(configs, minimum_cases=args.minimum_cases)
            print(f"已导出 {len(rows)} 条案例。")
        elif args.command == "analyze-ablation":
            detail_path, stats_path = analyze_position_ablation(configs)
            print(f"已输出配对明细 {detail_path} 和统计结果 {stats_path}。")
        elif args.command == "all":
            for config in configs:
                prepare_experiment(config)
                run_experiment(config, retry_failed=args.retry_failed, limit=args.limit)
                score_experiment(config, force=True)
            summaries = write_summaries(configs)
            write_figures(summaries, configs[0].paths.results_dir / "figures")
            export_cases(configs)
            print(f"已完成 {len(configs)} 个实验的全部流水线步骤。")
        return 0
    except (ConfigError, FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
