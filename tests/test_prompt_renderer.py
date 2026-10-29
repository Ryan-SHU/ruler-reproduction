import unittest

from ruler_repro.inference.prompt_renderer import render_author_prompt_template


class PromptRendererTest(unittest.TestCase):
    def test_meta_llama3_matches_upstream_special_token_order(self):
        templates = {
            "meta-llama3": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "{task_template}<|eot_id|><|start_header_id|>assistant"
                "<|end_header_id|>\n\n"
            )
        }
        rendered = render_author_prompt_template(
            templates,
            "meta-llama3",
            "TASK",
            " ANSWER",
        )
        self.assertEqual(
            rendered,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "TASK<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n ANSWER",
        )

    def test_base_preserves_existing_qwen_input(self):
        rendered = render_author_prompt_template(
            {"base": "{task_template}"},
            "base",
            "TASK",
            " ANSWER",
        )
        self.assertEqual(rendered, "TASK ANSWER")


if __name__ == "__main__":
    unittest.main()
