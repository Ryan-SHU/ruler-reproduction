from __future__ import annotations


class ExactMatchScorer:
    """只移除回答首尾空白后执行完整字符串比较。"""

    @staticmethod
    def normalize(prediction: str) -> str:
        return prediction.strip()

    def score(self, prediction: str, references: list[str]) -> bool:
        if not references:
            raise ValueError("标准答案列表不能为空。")
        expected = " ".join(references)
        return self.normalize(prediction) == expected
