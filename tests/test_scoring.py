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

    def test_variable_tracking_partial_score(self):
        scorer = RulerStringMatchScorer()
        references = ["AAAAA", "BBBBB", "CCCCC", "DDDDD", "EEEEE"]
        self.assertEqual(scorer.score_fraction("AAAAA, BBBBB, CCCCC", references), 0.6)
        self.assertFalse(scorer.score("AAAAA, BBBBB, CCCCC", references))

    def test_variable_tracking_strict_canonical_output(self):
        scorer = ExactMatchScorer()
        references = ["AAAAA", "BBBBB", "CCCCC"]
        self.assertTrue(scorer.score("AAAAA BBBBB CCCCC", references))
        self.assertFalse(scorer.score("AAAAA, BBBBB, CCCCC", references))


if __name__ == "__main__":
    unittest.main()
