from __future__ import annotations


class ExactMatchScorer:
    """只移除回答首尾空白后执行完整字符串比较。"""

    @staticmethod
    def normalize(prediction: str) -> str:
        return prediction.strip()

    def score(self, prediction: str, references: list[str]) -> bool:
        if len(references) != 1:
            raise ValueError("本项目严格匹配只接受唯一标准答案。")
        return self.normalize(prediction) == references[0]
