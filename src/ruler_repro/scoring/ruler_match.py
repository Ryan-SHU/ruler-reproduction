from __future__ import annotations


class RulerStringMatchScorer:
    """复现作者 string_match_all 对单样本的判断。"""

    def score(self, prediction: str, references: list[str]) -> bool:
        return self.score_fraction(prediction, references) == 1.0

    def score_fraction(self, prediction: str, references: list[str]) -> float:
        if not references:
            raise ValueError("标准答案列表不能为空。")
        lowered = prediction.lower()
        matched = sum(reference.lower() in lowered for reference in references)
        return matched / len(references)
