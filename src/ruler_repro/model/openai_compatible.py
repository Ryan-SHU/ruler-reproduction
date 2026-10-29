from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from ..schemas import ModelConfig, ModelRequest, ModelResponse, RetryConfig
from .base import ModelClient


class OpenAICompatibleClient(ModelClient):
    """调用本地 vLLM 的 OpenAI 兼容文本补全接口。"""

    def __init__(
        self,
        model: ModelConfig,
        retry: RetryConfig,
        event_logger: Callable[[str], None] | None = None,
    ):
        self.model = model
        self.retry = retry
        self.event_logger = event_logger

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.model.api_key_env:
            api_key = os.environ.get(self.model.api_key_env)
            if not api_key:
                raise RuntimeError(f"缺少 API 密钥环境变量：{self.model.api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _request_json(self, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=self._headers())
        method = "GET" if payload is None else "POST"
        request.method = method
        with urllib.request.urlopen(request, timeout=self.model.request_timeout_seconds) as response:
            body = response.read().decode("utf-8")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("服务响应顶层不是 JSON 对象。")
        return value

    def health_check(self) -> None:
        try:
            value = self._request_json(f"{self.model.base_url}/models")
        except Exception as exc:
            raise RuntimeError(f"无法访问模型服务 {self.model.base_url}：{exc}") from exc
        model_ids = {
            str(item.get("id")) for item in value.get("data", []) if isinstance(item, dict)
        }
        if self.model.model_name not in model_ids:
            raise RuntimeError(
                f"服务未暴露模型 {self.model.model_name}；当前模型为 {sorted(model_ids)}。"
            )

    def generate(self, request: ModelRequest, experiment_id: str) -> ModelResponse:
        payload = {
            "model": request.model_name,
            "prompt": request.prompt,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "seed": request.seed,
        }
        started = time.monotonic()
        final_status = "api_error"
        error_type: str | None = None
        error_message: str | None = None
        attempts = 0

        for attempts in range(1, self.retry.max_attempts + 1):
            transient = False
            try:
                value = self._request_json(f"{self.model.base_url}/completions", payload)
                choices = value.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise ValueError("响应缺少 choices[0]。")
                text = choices[0].get("text")
                if not isinstance(text, str):
                    raise ValueError("响应中的 choices[0].text 不是字符串。")
                usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
                return ModelResponse(
                    sample_id=request.sample_id,
                    experiment_id=experiment_id,
                    model_name=request.model_name,
                    raw_text=text,
                    status="success",
                    input_tokens=_optional_int(usage.get("prompt_tokens")),
                    output_tokens=_optional_int(usage.get("completion_tokens")),
                    latency_ms=round((time.monotonic() - started) * 1000, 3),
                    attempt_count=attempts,
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:1000]
                lower = body.lower()
                error_type = f"http_{exc.code}"
                error_message = body or str(exc)
                if exc.code == 429:
                    final_status, transient = "rate_limited", True
                elif exc.code in {500, 502, 503, 504}:
                    final_status, transient = "api_error", True
                elif exc.code in {401, 403}:
                    final_status = "authentication_error"
                elif "context" in lower and any(word in lower for word in ["length", "window", "token"]):
                    final_status = "context_overflow"
                else:
                    final_status = "api_error"
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                final_status, transient = "timeout", True
                error_type, error_message = type(exc).__name__, str(exc)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                final_status = "invalid_response"
                error_type, error_message = type(exc).__name__, str(exc)
            except Exception as exc:
                final_status = "api_error"
                error_type, error_message = type(exc).__name__, str(exc)

            if not transient or attempts >= self.retry.max_attempts:
                break
            wait_seconds = min(
                self.retry.initial_wait_seconds * (2 ** (attempts - 1)),
                self.retry.max_wait_seconds,
            )
            if self.event_logger is not None:
                self.event_logger(
                    f"sample_id={request.sample_id} retry_after_attempt={attempts} "
                    f"status={final_status} error_type={error_type} wait_seconds={wait_seconds}"
                )
            time.sleep(wait_seconds)

        return ModelResponse(
            sample_id=request.sample_id,
            experiment_id=experiment_id,
            model_name=request.model_name,
            raw_text=None,
            status=final_status,
            input_tokens=None,
            output_tokens=None,
            latency_ms=round((time.monotonic() - started) * 1000, 3),
            attempt_count=attempts,
            error_type=error_type,
            error_message=_sanitize_error(error_message),
        )


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _sanitize_error(message: str | None) -> str | None:
    if message is None:
        return None
    return message.replace("Authorization", "[已移除请求头]")[:1000]
