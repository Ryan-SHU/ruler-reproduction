from __future__ import annotations

from collections.abc import Mapping


class PromptTemplateError(ValueError):
    """模型输入模板缺失或格式无效。"""


def render_author_prompt_template(
    templates: Mapping[str, str],
    template_type: str,
    task_template: str,
    answer_prefix: str,
) -> str:
    """生成传给 RULER 作者生成器的完整模型模板。"""
    model_template = templates.get(template_type)
    if model_template is None:
        raise PromptTemplateError(f"上游模板中不存在 {template_type}。")
    if "{task_template}" not in model_template:
        raise PromptTemplateError(f"模板 {template_type} 缺少 task_template 占位符。")
    return model_template.format(task_template=task_template) + answer_prefix
