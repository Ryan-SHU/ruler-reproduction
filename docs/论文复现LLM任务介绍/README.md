# RULER 论文复现任务阅读索引

本目录介绍本项目实际复现的 RULER 任务与消融实验。RULER 是用合成数据评估长上下文语言模型的基准（benchmark，指统一任务、数据生成规则和评分方法的测试体系）。LLM 是大语言模型（Large Language Model）的缩写。合成数据指根据明确规则由程序自动生成的输入和标准答案。

阅读顺序如下：

1. [单针大海捞针任务](./01_单针大海捞针任务.md)：在长文本中检索一条键值证据。
2. [多键大海捞针任务](./02_多键大海捞针任务.md)：在多条结构相同的证据中根据目标键返回对应值。
3. [变量追踪任务](./03_变量追踪任务.md)：在长上下文中追踪多步变量赋值链。
4. [单针证据位置消融](./04_单针证据位置消融.md)：固定单针样本的其余内容，只改变证据所在位置。

前三份文档对应 RULER 的原始任务。第四份文档对应本项目为课程要求设计的单因素消融，其基础数据仍来自 RULER 单针任务。

主要依据为 RULER 论文第 3.1 节、第 3.2 节、附录 B 表 5、附录 F 表 14，以及作者仓库中的 `scripts/synthetic.yaml`、`scripts/data/synthetic/niah.py`、`scripts/data/synthetic/variable_tracking.py` 和 `scripts/eval/synthetic/constants.py`。
