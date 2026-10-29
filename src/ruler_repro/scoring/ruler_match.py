from __future__ import annotations


class RulerStringMatchScorer:
    """复现作者 string_match_all 对单样本的判断。"""

    def score(self, prediction: str, references: list[str]) -> bool:
        if not references:
            raise ValueError("标准答案列表不能为空。")
        lowered = prediction.lower()
        return all(reference.lower() in lowered for reference in references)
