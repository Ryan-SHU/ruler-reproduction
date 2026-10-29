import unittest

from ruler_repro.pipeline.evaluate import _classify_failure
from ruler_repro.schemas import ModelResponse, SampleRecord


def sample() -> SampleRecord:
    return SampleRecord(
        sample_id="test-42-0000",
        experiment_id="test",
        task_name="niah_multikey_1",
        target_length=2048,
        num_needle_k=4,
        input=(
            "One of the special magic numbers for alpha is: 1234567. "
            "One of the special magic numbers for beta is: 7654321."
        ),
        answer_prefix=" The special magic number for alpha mentioned in the provided text is",
        reference_answers=["1234567"],
        target_key="alpha",
        source_input_tokens=100,
        source_total_tokens=228,
        answer_token_position=10,
        random_seed=42,
        source_index=50,
    )


def response(text: str) -> ModelResponse:
    return ModelResponse(
        sample_id="test-42-0000",
        experiment_id="test",
        model_name="Qwen2.5-7B-Instruct",
        raw_text=text,
        status="success",
        input_tokens=100,
        output_tokens=2,
        latency_ms=10,
        attempt_count=1,
    )


class FailureClassificationTest(unittest.TestCase):
    def test_format_failure(self):
        category, _ = _classify_failure(sample(), response("Answer: 1234567"), "Answer: 1234567", False, True)
        self.assertEqual(category, "strict_format_failure")

    def test_distractor_confusion(self):
        category, _ = _classify_failure(sample(), response("7654321"), "7654321", False, False)
        self.assertEqual(category, "distractor_confusion")

    def test_empty_answer(self):
        category, _ = _classify_failure(sample(), response(""), "", False, False)
        self.assertEqual(category, "empty_answer")


if __name__ == "__main__":
    unittest.main()
