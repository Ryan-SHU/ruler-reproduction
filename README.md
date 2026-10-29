# RULER 课程复现代码

本目录是可直接分发给组员的完整实验工程。项目使用 RULER 作者生成器，分别用 `Qwen2.5-7B-Instruct` 与 `Meta-Llama-3.1-8B-Instruct` 的分词器和输入模板生成数据，再通过本地 `vLLM` 的 OpenAI 兼容接口完成推理、评分、汇总、绘图和失败案例导出。

实验范围包括单针与多键任务在 $2\mathrm{k}$、$4\mathrm{k}$、$8\mathrm{k}$ 下的主实验，以及单针 $8\mathrm{k}$、证据位置约 $10\%$、$50\%$、$90\%$ 的双模型消融。两个模型共享任务条件和评价流程，各自独立生成并处理数据。

开始操作前请完整阅读：

- [项目实施设计](docs/项目实施设计.md)
- [项目架构与复现流程 Step-by-Step](docs/项目架构与复现流程StepByStep.md)
- [运行与输出说明指南](docs/运行与输出说明指南.md)

最常用入口为 `ruler-repro`。执行 `ruler-repro --help` 可查看命令，执行 `python -m unittest discover -s tests -v` 可运行测试。模型权重、访问令牌和 Python 虚拟环境不得放入本目录。
