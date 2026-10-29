import os
import unittest
from pathlib import Path
from unittest.mock import patch

from ruler_repro.config import discover_configs, load_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    @patch.dict(os.environ, {"QWEN_MODEL_PATH": "/models/Qwen2.5-7B-Instruct"})
    def test_formal_config_and_discovery(self):
        config = load_experiment_config(PROJECT_ROOT / "configs/experiments/single_2k.yaml")
        self.assertEqual(config.max_seq_length, 2048)
        self.assertEqual(config.num_samples, 50)
        self.assertEqual(config.generation.max_output_tokens, 128)
        self.assertEqual(config.model.model_id, "qwen2_5_7b_instruct")
        configs = discover_configs(PROJECT_ROOT / "configs/experiments")
        self.assertEqual(len(configs), 7)

    @patch.dict(
        os.environ,
        {"LLAMA31_MODEL_PATH": "/models/Meta-Llama-3.1-8B-Instruct"},
    )
    def test_llama_override(self):
        config = load_experiment_config(
            PROJECT_ROOT / "configs/experiments/single_2k.yaml",
            PROJECT_ROOT / "configs/models/llama3_1_8b_instruct.yaml",
        )
        self.assertEqual(config.model.model_id, "llama3_1_8b_instruct")
        self.assertEqual(config.model.prompt_template_type, "meta-llama3")
        self.assertEqual(config.model.context_window, 131072)

    @patch.dict(os.environ, {"QWEN_MODEL_PATH": "/models/Qwen2.5-7B-Instruct"})
    def test_ablation_config(self):
        config = load_experiment_config(
            PROJECT_ROOT / "configs/ablations/position_single_8k_p50.yaml"
        )
        self.assertEqual(config.ablation_factor, "evidence_position")
        self.assertEqual(config.evidence_position_percent, 50)
        self.assertEqual(config.base_experiment_id, "single_8k")


if __name__ == "__main__":
    unittest.main()
