import unittest

from .two_moons import Scorer


class ScorerTestCase(unittest.TestCase):
    def test_scores_to_overall(self):
        scores = {
            "Which_MOON": 91.0,
        }

        overall = Scorer.scores_to_overall(scores)
        self.assertAlmostEqual(overall["TWO_MOONS"], 91.0)
        self.assertAlmostEqual(overall["Overall"], 91.0)

    def test_scores_to_metrics(self):
        scores = {
            "Which_MOON": 91.0,
        }

        metrics = Scorer.scores_to_metrics(scores)
        self.assertAlmostEqual(metrics["Which_MOON"], 91.0)

    def test_get_metric_names(self):
        metrics = Scorer.get_metric_names()
        self.assertEqual(
            metrics,
            [
                "Which_MOON",
            ],
        )

    def test_get_maximize_metrics(self):
        self.assertEqual(
            set(Scorer.get_overall_names()), Scorer.get_maximize_metrics().keys()
        )
