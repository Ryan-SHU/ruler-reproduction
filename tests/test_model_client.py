import unittest
from unittest.mock import patch

from ruler_repro.model.openai_compatible import OpenAICompatibleClient
from ruler_repro.schemas import ModelConfig, ModelRequest, RetryConfig


class StubClient(OpenAICompatibleClient):
    def __init__(self, values, events):
        super().__init__(
            ModelConfig(
                model_id="qwen2_5_7b_instruct",
                model_name="Qwen2.5-7B-Instruct",
                base_url="http://local/v1",
                tokenizer_path="/model",
            ),
            RetryConfig(max_attempts=3, initial_wait_seconds=0, max_wait_seconds=0),
            event_logger=events.append,
        )
        self.values = iter(values)

    def _request_json(self, url, payload=None):
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value


class ModelClientTest(unittest.TestCase):
    @patch("ruler_repro.model.openai_compatible.time.sleep", return_value=None)
    def test_transient_failure_is_bounded_and_recorded(self, _sleep):
        import urllib.error

        events = []
        client = StubClient(
            [
                urllib.error.URLError("temporary"),
                {"choices": [{"text": "1234567"}], "usage": {"prompt_tokens": 100, "completion_tokens": 2}},
            ],
            events,
        )
        request = ModelRequest("s", "Qwen2.5-7B-Instruct", "prompt", 0, 1, 128, 42)
        response = client.generate(request, "exp")
        self.assertEqual(response.status, "success")
        self.assertEqual(response.attempt_count, 2)
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
