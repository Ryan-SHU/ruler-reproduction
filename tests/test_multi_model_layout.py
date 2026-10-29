import os
import unittest
from pathlib import Path
from unittest.mock import patch

from ruler_repro.config import load_experiment_config
from ruler_repro.pipeline.infer import _paths
from ruler_repro.pipeline.summarize import _wilson_ci
from ruler_repro.schemas import SampleRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MultiModelLayoutTest(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "QWEN_MODEL_PATH": "/models/qwen",
            "LLAMA31_MODEL_PATH": "/models/llama",
        },
    )
    def test_main_and_ablation_cell_counts_and_path_isolation(self):
        model_paths = [
            PROJECT_ROOT / "configs/models/qwen2_5_7b_instruct.yaml",
            PROJECT_ROOT / "configs/models/llama3_1_8b_instruct.yaml",
        ]
        main_paths = sorted((PROJECT_ROOT / "configs/experiments").glob("*.yaml"))
        ablation_paths = sorted((PROJECT_ROOT / "configs/ablations").glob("*.yaml"))
        main = [load_experiment_config(path, model) for model in model_paths for path in main_paths]
        ablation = [load_experiment_config(path, model) for model in model_paths for path in ablation_paths]
        self.assertEqual(len(main), 14)
        self.assertEqual(len(ablation), 6)
        qwen_response = _paths(main[0])[1]
        llama_response = _paths(main[7])[1]
        self.assertNotEqual(qwen_response, llama_response)
        self.assertIn("qwen2_5_7b_instruct", str(qwen_response))
        self.assertIn("llama3_1_8b_instruct", str(llama_response))

    def test_legacy_sample_fields_remain_readable(self):
        value = {
            "sample_id": "legacy-0",
            "experiment_id": "single_2k",
            "task_name": "niah_single_2",
            "target_length": 2048,
            "num_needle_k": 1,
            "input": "One of the special magic numbers for alpha is: 1234567.",
            "answer_prefix": " answer",
            "reference_answers": ["1234567"],
            "target_key": "alpha",
            "source_input_tokens": 100,
            "source_total_tokens": 228,
            "answer_token_position": 10,
            "random_seed": 42,
            "source_index": 0,
        }
        sample = SampleRecord.from_dict(value)
        self.assertEqual(sample.model_id, "legacy")
        self.assertIsNone(sample.base_sample_id)

    def test_wilson_interval_has_valid_bounds(self):
        low, high = _wilson_ci(25, 50)
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertLess(low, 50)
        self.assertGreater(high, 50)


if __name__ == "__main__":
    unittest.main()
