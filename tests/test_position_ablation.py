import os
import unittest
from pathlib import Path
from unittest.mock import patch

from ruler_repro.config import load_experiment_config
from ruler_repro.data.position_ablation import _relocate
from ruler_repro.schemas import SampleRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    def tokenize(self, text):
        return text.split()


class PositionAblationTest(unittest.TestCase):
    @patch.dict(os.environ, {"QWEN_MODEL_PATH": "/models/Qwen2.5-7B-Instruct"})
    @patch(
        "ruler_repro.data.position_ablation._sent_tokenize",
        side_effect=lambda text: [part.strip() + "." for part in text.split(".") if part.strip()],
    )
    def test_relocate_preserves_answer_and_records_pair(self, _):
        config = load_experiment_config(
            PROJECT_ROOT / "configs/ablations/position_single_8k_p50.yaml"
        )
        sample = SampleRecord(
            sample_id="base",
            experiment_id="single_8k",
            task_name="niah_single_2",
            target_length=8192,
            num_needle_k=1,
            input=(
                "A special magic number is hidden within the following text. "
                "Make sure to memorize it. I will quiz you about the number afterwards.\n"
                "First sentence. One of the special magic numbers for alpha is: 1234567. "
                "Second sentence. Third sentence.\n"
                "What is the special magic number for alpha mentioned in the provided text?"
            ),
            answer_prefix=" The special magic number for alpha mentioned in the provided text is",
            reference_answers=["1234567"],
            target_key="alpha",
            source_input_tokens=80,
            source_total_tokens=208,
            answer_token_position=20,
            random_seed=42,
            source_index=0,
            model_id="qwen2_5_7b_instruct",
        )
        result = _relocate(sample, config, FakeTokenizer())
        self.assertEqual(result.base_sample_id, "base")
        self.assertEqual(result.evidence_position_target_percent, 50)
        self.assertEqual(result.input.count("1234567"), 1)
        self.assertEqual(result.reference_answers, ["1234567"])


if __name__ == "__main__":
    unittest.main()
