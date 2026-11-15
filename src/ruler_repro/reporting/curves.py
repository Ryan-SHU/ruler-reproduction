from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..schemas import ExperimentSummary


def write_figures(summaries: list[ExperimentSummary], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    main = [item for item in summaries if not item.experiment_id.startswith("position_")]
    ablation = [item for item in summaries if item.experiment_id.startswith("position_")]
    created.extend(_length_curve(main, "niah_single_2", output_dir / "single_length_curve_by_model.png"))
    created.extend(_length_curve(main, "niah_multikey_1", output_dir / "multikey_length_curve_by_model.png", needles=4))
    created.extend(_needle_curve(main, output_dir / "multikey_8k_needle_count_by_model.png"))
    created.extend(_position_curve(ablation, output_dir / "position_ablation_by_model.png"))
    if not any(item.task_name == "vt" and item.num_hops == 2 for item in main):
        created.extend(_variable_tracking_curve(main, output_dir / "variable_tracking_length_curve_by_model.png"))
    created.extend(_vt_hops_curve(main, output_dir / "vt_hops_ablation_by_model.png"))
    return created


def _variable_tracking_curve(
    summaries: list[ExperimentSummary],
    path: Path,
) -> list[Path]:
    selected = [
        item for item in summaries
        if item.task_name == "vt" and item.num_hops == 4 and item.ruler_score_mean is not None
    ]
    selected.sort(key=lambda item: (item.model_id, item.target_length))
    if not selected:
        return []
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    for model_id in sorted({item.model_id for item in selected}):
        group = [item for item in selected if item.model_id == model_id]
        axis.plot(
            [item.target_length for item in group],
            [item.ruler_score_mean for item in group],
            marker="o",
            label=model_id,
        )
    axis.legend()
    axis.set_xlabel("Target sequence budget (tokens)")
    axis.set_ylabel("RULER variable recall score (%)")
    axis.set_xticks([8192, 16384, 32768], ["8k", "16k", "32k"])
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [path]


def _vt_hops_curve(summaries: list[ExperimentSummary], path: Path) -> list[Path]:
    selected = [
        item for item in summaries
        if item.task_name == "vt" and item.target_length == 32768
        and item.num_hops in {2, 4} and item.ruler_score_mean is not None
    ]
    if not any(item.num_hops == 2 for item in selected):
        return []
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    for model_id in sorted({item.model_id for item in selected}):
        group = sorted((item for item in selected if item.model_id == model_id), key=lambda item: item.num_hops)
        axis.plot([item.num_hops for item in group], [item.ruler_score_mean for item in group], marker="o", label=model_id)
    axis.set_xlabel("Number of assignment hops")
    axis.set_ylabel("RULER variable recall score (%)")
    axis.set_xticks([2, 4])
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [path]


def _length_curve(
    summaries: list[ExperimentSummary],
    task_name: str,
    path: Path,
    needles: int | None = None,
) -> list[Path]:
    selected = [
        item for item in summaries
        if item.task_name == task_name
        and (needles is None or item.num_needle_k == needles)
        and item.exact_match_rate is not None
    ]
    selected.sort(key=lambda item: (item.model_id, item.target_length))
    if not selected:
        return []
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    for model_id in sorted({item.model_id for item in selected}):
        group = [item for item in selected if item.model_id == model_id]
        axis.plot(
            [item.target_length for item in group],
            [item.exact_match_rate for item in group],
            marker="o",
            label=model_id,
        )
    axis.legend()
    axis.set_xlabel("Target sequence budget (tokens)")
    axis.set_ylabel("Exact match (%)")
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [path]


def _needle_curve(summaries: list[ExperimentSummary], path: Path) -> list[Path]:
    selected = [
        item for item in summaries
        if item.task_name == "niah_multikey_1"
        and item.target_length == 8192
        and item.num_needle_k in {4, 8}
        and item.exact_match_rate is not None
    ]
    selected.sort(key=lambda item: (item.model_id, item.num_needle_k))
    if not selected:
        return []
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    model_ids = sorted({item.model_id for item in selected})
    needle_values = [4, 8]
    width = 0.35
    for model_index, model_id in enumerate(model_ids):
        group = {item.num_needle_k: item for item in selected if item.model_id == model_id}
        offsets = [index + (model_index - (len(model_ids) - 1) / 2) * width for index in range(2)]
        axis.bar(offsets, [group[n].exact_match_rate if n in group else 0 for n in needle_values], width, label=model_id)
    axis.set_xticks(range(2), [str(value) for value in needle_values])
    axis.legend()
    axis.set_xlabel("Number of key-value needles")
    axis.set_ylabel("Exact match (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [path]


def _position_curve(summaries: list[ExperimentSummary], path: Path) -> list[Path]:
    selected = [item for item in summaries if item.exact_match_rate is not None]
    if not selected:
        return []
    def position(item: ExperimentSummary) -> int:
        return int(item.experiment_id.rsplit("p", 1)[1])
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    for model_id in sorted({item.model_id for item in selected}):
        group = sorted(
            [item for item in selected if item.model_id == model_id],
            key=position,
        )
        axis.plot(
            [position(item) for item in group],
            [item.exact_match_rate for item in group],
            marker="o",
            label=model_id,
        )
    axis.set_xlabel("Evidence position target (%)")
    axis.set_ylabel("Exact match (%)")
    axis.set_xticks([10, 50, 90])
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [path]
