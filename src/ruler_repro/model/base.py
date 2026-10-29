from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import ModelRequest, ModelResponse


class ModelClient(ABC):
    @abstractmethod
    def health_check(self) -> None:
        """确认服务可访问且模型名称存在。"""

    @abstractmethod
    def generate(self, request: ModelRequest, experiment_id: str) -> ModelResponse:
        """执行一次带有限重试的生成请求。"""
