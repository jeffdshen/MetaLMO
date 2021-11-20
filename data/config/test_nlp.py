import unittest

from .nlp import Scorer


class ScorerTestCase(unittest.TestCase):
    def test_scores_to_overall(self):
        scores = {
            "BoolQ": 91.0,
            "CB": (98.6, 99.2),
            "COPA": 97.4,
            "MultiRC": (88.6, 63.2),
            "ReCoRD": (94.7, 94.2),
            "RTE": 92.6,
            "WiC": 77.4,
            "WSC": 97.3,
        }

        overall = Scorer.scores_to_overall(scores)
        self.assertAlmostEqual(overall["SuperGLUE"], 90.61875)
        self.assertAlmostEqual(overall["Overall"], 90.61875)

    def test_scores_to_metrics(self):
        scores = {
            "BoolQ": 91.0,
            "CB": (98.6, 99.2),
            "COPA": 97.4,
            "MultiRC": (88.6, 63.2),
            "ReCoRD": (94.7, 94.2),
            "RTE": 92.6,
            "WiC": 77.4,
            "WSC": 97.3,
        }
        metrics = Scorer.scores_to_metrics(scores)
        self.assertAlmostEqual(metrics["BoolQ"], 91.0)
        self.assertAlmostEqual(metrics["CB_F1"], 98.6)
        self.assertAlmostEqual(metrics["CB_Acc"], 99.2)
        self.assertAlmostEqual(metrics["COPA"], 97.4)
        self.assertAlmostEqual(metrics["MultiRC_F1a"], 88.6)
        self.assertAlmostEqual(metrics["MultiRC_EM"], 63.2)
        self.assertAlmostEqual(metrics["ReCoRD_F1"], 94.7)
        self.assertAlmostEqual(metrics["ReCoRD_Acc"], 94.2)
        self.assertAlmostEqual(metrics["RTE"], 92.6)
        self.assertAlmostEqual(metrics["WiC"], 77.4)
        self.assertAlmostEqual(metrics["WSC"], 97.3)

    def test_get_metric_names(self):
        metrics = Scorer.get_metric_names()
        self.assertEqual(
            metrics,
            [
                "BoolQ",
                "CB_F1",
                "CB_Acc",
                "COPA",
                "MultiRC_F1a",
                "MultiRC_EM",
                "ReCoRD_F1",
                "ReCoRD_Acc",
                "RTE",
                "WiC",
                "WSC",
            ],
        )

        metrics = Scorer.get_metric_names(["MultiRC", "BoolQ", "CB"])
        self.assertEqual(
            metrics, ["MultiRC_F1a", "MultiRC_EM", "BoolQ", "CB_F1", "CB_Acc"]
        )

    def test_get_maximize_metrics(self):
        self.assertEqual(
            set(Scorer.get_overall_names()), Scorer.get_maximize_metrics().keys()
        )
