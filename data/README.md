# 实验数据说明

本目录只保存本项目运行时产生的数据索引与样本，不保存模型权重。`generated/<model_id>/<experiment_id>/source.jsonl` 是 RULER 作者生成器的原始输出，`manifests/<model_id>/<experiment_id>/samples.jsonl` 是经过字段统一和完整性检查后的冻结样本清单。

Qwen 与 Llama 使用相同的任务条件和固定随机种子，但分别使用自己的分词器与 RULER 输入模板生成数据。两个模型的数据按 `model_id` 隔离，不能相互替换。`configs/ablations` 中的位置消融从同一模型的 `single_8k` 样本生成，并以 `base_sample_id` 保存三个位置变体之间的配对关系。

样本背景来自 RULER 上游脚本下载的 Paul Graham Essays，原始背景文件位于 `../../ref/RULER/scripts/data/synthetic/json/PaulGrahamEssays.json`。正式模型调用只读取 `manifests` 中的冻结清单；生成命令、数据字段与复核方式见[运行与输出说明指南](../docs/运行与输出说明指南.md)。
