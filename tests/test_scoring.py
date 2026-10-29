import unittest

from ruler_repro.scoring.exact_match import ExactMatchScorer
from ruler_repro.scoring.ruler_match import RulerStringMatchScorer


class ScoringTest(unittest.TestCase):
    def test_exact_match_only_strips_outer_whitespace(self):
        scorer = ExactMatchScorer()
        self.assertTrue(scorer.score(" 1234567\n", ["1234567"]))
        self.assertFalse(scorer.score("Answer: 1234567", ["1234567"]))

    def test_ruler_match_accepts_reference_substring(self):
        scorer = RulerStringMatchScorer()
        self.assertTrue(scorer.score("Answer: 1234567", ["1234567"]))
        self.assertFalse(scorer.score("7654321", ["1234567"]))


if __name__ == "__main__":
    unittest.main()
