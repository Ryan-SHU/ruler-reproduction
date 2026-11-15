# RULER Reproduction | RULER 长上下文评测复现

[中文](#中文) · [English](#english)

## 中文

### 项目简介

本项目复现 [RULER](https://github.com/NVIDIA/RULER) 长上下文评测中的部分合成任务，使用作者的数据生成器，分别评测 Qwen2.5-7B-Instruct 和 Meta-Llama-3.1-8B-Instruct。工程提供样本生成、模型调用、逐条评分、汇总绘图与失败案例导出的完整流程。两个模型各自使用对应的分词器和输入模板生成样本。

本项目是课程论文复现工程。所测任务和长度见下表；实验结论应依据实际运行结果，并限定在这些条件内。

| 任务 | RULER 配置 | 本项目条件 | 每个模型每格样本数 |
| --- | --- | --- | ---: |
| 单针检索 | `niah_single_2` | $2\mathrm{k}$、$4\mathrm{k}$、$8\mathrm{k}$ | $50$ |
| 多键检索 | `niah_multikey_1` | 四键：$2\mathrm{k}$、$4\mathrm{k}$、$8\mathrm{k}$；八键：$8\mathrm{k}$ | $50$ |
| 变量追踪 | `vt` | $8\mathrm{k}$、$16\mathrm{k}$、$32\mathrm{k}$；一条链、四跳 | $50$ |
| 单针证据位置消融 | 基于 `niah_single_2` | $8\mathrm{k}$；目标位置 $10\%$、$50\%$、$90\%$ | $50$ |
| VT 跳数消融 | 基于 `vt` | $32\mathrm{k}$；两跳与四跳对照 | $50$ |

这里的长度是输入与预留输出合计的词元（token）预算。项目同时报告严格答案匹配、RULER 作者评分、实际输入词元长度和调用量。变量追踪还记录五个目标变量全部命中的比例及逐变量平均命中比例。评分定义见[任务介绍](docs/论文复现LLM任务介绍/README.md)。

### 仓库结构

```text
configs/             模型、任务和实验条件配置
src/ruler_repro/      数据生成适配、推理、评分与报告代码
tests/                自动测试
docs/                 设计、任务介绍与逐步运行指南
data/                 运行时生成的样本目录（内容不纳入 Git）
results/              运行时生成的响应和分析目录（内容不纳入 Git）
```

作者的 RULER 代码位于本仓库之外的 `../ref/RULER`；模型权重也由使用者单独下载。安装与运行前，请先阅读[完整运行与输出说明指南](docs/运行与输出说明指南.md)，其中包含上游代码、背景数据、模型权重和本地模型服务的准备步骤。

### 快速开始

以下命令在本仓库根目录执行。首先按运行指南准备 `../ref/RULER`、Paul Graham Essays、两个模型的本地权重，以及兼容 OpenAI 接口的本地 vLLM 服务。模型路径通过 `QWEN_MODEL_PATH` 和 `LLAMA31_MODEL_PATH` 环境变量配置；模型服务名称须与 `configs/models/` 中的配置一致。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v

ruler-repro preflight \
  --config configs/pilot/pilot_single_2k.yaml \
  --model-config configs/models/qwen2_5_7b_instruct.yaml \
  --check-server

ruler-repro all \
  --config-dir configs/pilot \
  --include-pilot \
  --model-config configs/models/qwen2_5_7b_instruct.yaml
```

这组命令执行 Qwen 的小范围流程检查。正式实验、Llama 服务切换、变量追踪与结果合并的逐步命令见[运行与输出说明指南](docs/运行与输出说明指南.md)。`ruler-repro --help` 可查看命令入口。

### 输出与复现资料

运行产生的样本清单、原始响应和评分分别位于 `data/manifests/`、`results/raw/` 和 `results/scored/`，均按模型及实验编号分目录保存。汇总表、图和案例分别位于 `results/summary/`、`results/figures/` 和 `results/cases/`。这些运行产物由 `.gitignore` 排除，组内共享时需单独传递。

- [项目实施设计](docs/项目实施设计.md)：模块、接口和数据流。
- [项目架构与复现流程](docs/项目架构与复现流程StepByStep.md)：各文件的职责与完整流程。
- [任务介绍](docs/论文复现LLM任务介绍/README.md)：任务生成和评分规则。
- [运行与输出说明指南](docs/运行与输出说明指南.md)：环境准备、逐阶段命令和输出检查。

### 来源与许可

本仓库自己的代码与文档采用 [MIT 许可证](LICENSE)。数据生成调用 NVIDIA 的 [RULER 作者代码](https://github.com/NVIDIA/RULER)，上游代码采用 Apache-2.0 许可证。模型权重和背景文章遵循各自来源的使用条款。

论文引用：Hsieh et al., *RULER: What's the Real Context Size of Your Long-Context Language Models?*, COLM 2024。论文及引用信息见[作者仓库](https://github.com/NVIDIA/RULER)和[论文页面](https://arxiv.org/abs/2404.06654)。

## English

### Overview

This course project reproduces selected synthetic tasks from [RULER](https://github.com/NVIDIA/RULER). It uses the authors' generators to evaluate Qwen2.5-7B-Instruct and Meta-Llama-3.1-8B-Instruct, with separate tokenizers and prompt templates for each model. The pipeline covers sample preparation, local model inference, per-sample scoring, summaries, plots, and failure-case exports.

The evaluated scope is fixed. Interpret results for the models, tasks, and lengths actually run.

| Task | RULER configuration | Conditions in this project | Samples per model and condition |
| --- | --- | --- | ---: |
| Single-needle retrieval | `niah_single_2` | $2\mathrm{k}$, $4\mathrm{k}$, $8\mathrm{k}$ | $50$ |
| Multi-key retrieval | `niah_multikey_1` | Four keys at $2\mathrm{k}$, $4\mathrm{k}$, $8\mathrm{k}$; eight keys at $8\mathrm{k}$ | $50$ |
| Variable tracking | `vt` | $8\mathrm{k}$, $16\mathrm{k}$, $32\mathrm{k}$; one chain and four hops | $50$ |
| Evidence-position ablation | Based on `niah_single_2` | $8\mathrm{k}$; target positions $10\%$, $50\%$, $90\%$ | $50$ |
| VT hop-count ablation | Based on `vt` | Two versus four hops at $32\mathrm{k}$ | $50$ |

Lengths denote the combined input and reserved output token budget. Reported measures include strict answer match, the original RULER score, actual input-token length, and call counts. Variable tracking also reports full-chain match and mean per-variable recall.

### Repository layout

```text
configs/             Model and experiment configurations
src/ruler_repro/      Data adapters, inference, scoring, and reporting
tests/                Automated tests
docs/                 Design, task explanations, and run guide
data/                 Generated samples (ignored by Git)
results/              Responses and analyses (ignored by Git)
```

The upstream RULER repository is expected at `../ref/RULER`. Model weights are downloaded separately. See the [complete run and output guide (Chinese)](docs/运行与输出说明指南.md) for upstream data, models, service setup, and commands for every project stage.

### Quick start

Run these commands from this repository's root after preparing the upstream repository, Paul Graham Essays, local model weights, and an OpenAI-compatible vLLM server. Set `QWEN_MODEL_PATH` and `LLAMA31_MODEL_PATH` as described in the run guide; the served model name must match `configs/models/`.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v

ruler-repro preflight \
  --config configs/pilot/pilot_single_2k.yaml \
  --model-config configs/models/qwen2_5_7b_instruct.yaml \
  --check-server

ruler-repro all \
  --config-dir configs/pilot \
  --include-pilot \
  --model-config configs/models/qwen2_5_7b_instruct.yaml
```

This runs a small Qwen pipeline check. The [run guide](docs/运行与输出说明指南.md) provides the full commands for both models, formal experiments, variable tracking, and result consolidation. Use `ruler-repro --help` for the command interface.

### Outputs and documentation

Samples, raw responses, and scores are stored under `data/manifests/`, `results/raw/`, and `results/scored/`, organized by model and experiment. Tables, figures, and case exports are under `results/summary/`, `results/figures/`, and `results/cases/`. Generated artifacts are ignored by Git.

- [Implementation design (Chinese)](docs/项目实施设计.md)
- [Architecture and workflow (Chinese)](docs/项目架构与复现流程StepByStep.md)
- [Task descriptions (Chinese)](docs/论文复现LLM任务介绍/README.md)
- [Run and output guide (Chinese)](docs/运行与输出说明指南.md)

### Attribution and licensing

This repository's own code and documentation are licensed under the [MIT License](LICENSE). Data generation uses the [original NVIDIA RULER repository](https://github.com/NVIDIA/RULER), which is licensed under Apache-2.0. Model weights and source essays remain subject to their respective terms.

Paper: Hsieh et al., *RULER: What's the Real Context Size of Your Long-Context Language Models?*, COLM 2024. See the [authors' repository](https://github.com/NVIDIA/RULER) and [paper](https://arxiv.org/abs/2404.06654) for citation details.
