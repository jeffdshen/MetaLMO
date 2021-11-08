import unittest

import torch
import torch.nn.functional as F

from .tasks import scores_to_metrics, get_tasks, scores_to_overall
from .tokenizers import get_fake_tokenizer


def setUpModule():
    global tokenizer
    tokenizer = get_fake_tokenizer(
        [
            "The quick brown fox jumps over the lazy dog!",
            "Does the slow green dog slide under the running fox?",
            "The green fox slides under the slow dog!",
            "Does the jumping brown fox runs over the quickly lazing dog?",
        ],
        vocab_size=512,
        cls_count=30,
        model_max_length=24,
        stride=8,
    )
    global examples
    examples = [
        (
            "The quick brown dog jumped over the lazy fox.",
            "Did brown dog jump over fox?",
        ),
        (
            "The quick brown dog jumped over the lazy fox.",
            "Did green dog jump over fox?",
        ),
        ("The green dog slides over the slow fox.", "Does dog run over fox?"),
    ]


def mark_seen(s, t, seen):
    k = s.index(t)
    for i in range(k, k + len(t)):
        seen.add(i)


def split_tokens(tokens, sep_id):
    k = tokens.index(sep_id)
    return tokens[1:k], tokens[k:]


def assert_features(self, features, window, first, second):
    self.assertTorchScalarEqual(features[:, 0], self.task.task_id)
    self.assertTorchScalarEqual(torch.sum(features == self.sep_id, dim=1), 2)

    p_seen = set()
    q_seen = set()
    for tokens in features.tolist():
        q, p = split_tokens(tokens, self.sep_id)
        q = tokenizer.decode(q)
        p = tokenizer.decode(p)
        self.assertIn(q, first)
        self.assertIn(p, second)
        mark_seen(first, q, q_seen)
        mark_seen(second, p, p_seen)

    if window == "all":
        self.assertEqual(q_seen, set(range(len(first))))
        self.assertEqual(p_seen, set(range(len(second))))

    if window == "first":
        self.assertIn(0, q_seen)
        self.assertIn(0, p_seen)


class BoolQTest(unittest.TestCase):
    def setUp(self):
        self.task = get_tasks(tokenizer)["BoolQ"]
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.examples = [
            {
                "idx": 0,
                "label": True,
                "passage": examples[0][0],
                "question": examples[0][1],
            },
            {
                "idx": 1,
                "label": False,
                "passage": examples[1][0],
                "question": examples[1][1],
            },
            {
                "idx": 2,
                "label": False,
                "passage": examples[2][0],
                "question": examples[2][1],
            },
        ]

    def assertTorchEqual(self, a, b):
        self.assertEqual(a.tolist(), b.tolist())

    def assertTorchScalarEqual(self, a, b):
        self.assertTorchEqual(a, torch.full_like(a, b))

    def test_encode(self):
        for example in self.examples:
            for window in ["all", "random", "first"]:
                idxs, features, labels = self.task.encode(example, window=window)

                self.assertEqual(idxs.size(0), labels.size(0))
                self.assertEqual(labels.size(), features.size())
                if window in ["random", "first"]:
                    self.assertEqual(idxs.size(0), 1)

                self.assertTorchScalarEqual(idxs, example["idx"])
                self.assertTorchScalarEqual(
                    labels, self.task.labels.to_token(example["label"])
                )

                assert_features(
                    self, features, window, example["question"], example["passage"]
                )

    def test_predict(self):
        idxs = [0, 0, 1, 1, 2]
        features = [[4, 2], [2, 2], [4, 2], [4, 0], [1, 0]]
        outputs = torch.full((5, 2, 512), 0.0, dtype=torch.float32)

        idxs = torch.tensor(idxs, dtype=torch.long)
        features = torch.tensor(features, dtype=torch.long)
        outputs[:, :, 4] = 1.0
        outputs[:, :, 2] = 1.0

        # test average
        outputs[0, 0, 4] += 1.0
        outputs[1, 1, 2] += 2.0
        outputs[2, 0, 2] += 1.1
        outputs[3, 0, 4] += 1.0
        outputs[4, 0, 4] += 1.0

        # padding and non-classes don't matter
        outputs[:, :, 3] = 10.0
        outputs[4, 1, 4] = 10.0
        outputs[3, 1, 2] = 10.0

        # constant shift doesn't matter
        outputs[1, 0, :] += 100.0

        self.assertEqual(
            self.task.predict(idxs, features, outputs), {0: False, 1: False, 2: True}
        )

        # answer doesn't change even if we do log_softmax
        outputs = F.log_softmax(outputs, dim=-1)
        self.assertEqual(
            self.task.predict(idxs, features, outputs), {0: False, 1: False, 2: True}
        )

    def test_score(self):
        pred = {0: False, 1: False, 2: True}
        examples = [
            {"idx": 0, "label": False},
            {"idx": 1, "label": True},
            {"idx": 2, "label": True},
            {"idx": 3, "label": True},
        ]
        self.assertEqual(self.task.score(pred, examples, strict=False), 2 / 3)


class ReCoRDTestCase(unittest.TestCase):
    def setUp(self):
        self.task = get_tasks(tokenizer)["ReCoRD"]
        self.sep_id = tokenizer.token_to_id("[SEP]")

        self.examples = [
            {
                "passage": {
                    "idx": 0,
                    "entities": [
                        {"start": 16, "end": 18},
                        {"start": 41, "end": 43},
                    ],
                    "text": examples[0][0],
                    "highlight": "The quick brown *dog* jumped over the lazy *fox*.",
                },
                "qas": [
                    {
                        "idx": 0,
                        "query": examples[0][1],
                        "answers": [
                            {
                                "start": 16,
                                "end": 18,
                                "text": "dog",
                            }
                        ],
                    },
                    {
                        "idx": 1,
                        "query": examples[1][1],
                        "answers": [
                            {
                                "start": 16,
                                "end": 18,
                                "text": "dog",
                            },
                            {
                                "start": 41,
                                "end": 43,
                                "text": "fox",
                            },
                        ],
                    },
                ],
            },
            {
                "passage": {
                    "idx": 1,
                    "entities": [
                        {"start": 4, "end": 8},
                        {"start": 14, "end": 19},
                        {"start": 30, "end": 33},
                    ],
                    "text": examples[2][0],
                    "highlight": "The *green* dog *slides* over the *slow* fox.",
                },
                "qas": [
                    {
                        "idx": 2,
                        "query": examples[2][1],
                        "answers": [
                            {
                                "start": 14,
                                "end": 19,
                                "text": "slides",
                            }
                        ],
                    },
                ],
            },
        ]

    def assertTorchEqual(self, a, b):
        self.assertEqual(a.tolist(), b.tolist())

    def assertTorchScalarEqual(self, a, b):
        self.assertTorchEqual(a, torch.full_like(a, b))

    def test_encode(self):
        for orig_example in self.examples:
            example = orig_example.copy()
            for question in orig_example["qas"]:
                example["qas"] = question

                for window in ["all", "random", "first"]:
                    idxs, features, labels = self.task.encode(example, window=window)

                    self.assertEqual(idxs.size(0), labels.size(0))
                    self.assertEqual(labels.size(), features.size())
                    self.assertTorchScalarEqual(idxs, example["qas"]["idx"])
                    if window in ["random", "first"]:
                        self.assertEqual(idxs.size(0), 1)

                    self.assertTorchScalarEqual(
                        (labels == features).logical_or(labels == self.sep_id), True
                    )
                    for feature, label in zip(features, labels):
                        label = tokenizer.decode(label[label != self.sep_id].tolist())
                        _, passage = split_tokens(feature.tolist(), self.sep_id)

                        passage = tokenizer.decode(passage)
                        self.assertEqual(
                            label,
                            "".join(
                                a["text"]
                                for a in example["qas"]["answers"]
                                if a["text"] in passage
                            ),
                        )

                    assert_features(
                        self,
                        features,
                        window,
                        example["qas"]["query"],
                        example["passage"]["highlight"],
                    )

    def test_predict(self):
        idxs = [0, 0, 1, 2]
        features = [
            [306, 253, 42, 313, 335, 42, 301, 253, 0],
            [253, 42, 296, 314, 115, 42, 340, 291, 253],
            [42, 296, 295, 42, 302, 46, 0, 0, 0],
            [291, 253, 42, 296, 295, 42, 302, 46, 0],
        ]
        outputs = torch.full((4, 9, 512), 0.0, dtype=torch.float32)
        idxs = torch.tensor(idxs, dtype=torch.long)
        features = torch.tensor(features, dtype=torch.long)
        outputs[:, :, 2] = 1.0

        # test max span
        outputs[0, 3, 313] += 2.0
        outputs[0, 4, 335] += 4.0
        outputs[1, 2, 296] += 3.0
        outputs[1, 3, 314] += 3.0
        outputs[1, 4, 290] += 3.0
        outputs[2, 1, 296] += 5.0
        outputs[2, 2, 295] += 2.0
        outputs[2, 2, 296] += 1.0
        outputs[3, 3, 296] += 2.0
        outputs[3, 4, 295] += 3.0
        outputs[3, 6, 302] += 3.0

        # padding doesn't matter
        outputs[0, 8, :] = 10.0
        outputs[2, 6, 290] = 10.0
        outputs[2, 7, :] = 10.0
        outputs[3, 8, 290] = 10.0

        # constant shift doesn't matter
        outputs[0, 0, :] += 100.0
        outputs[0, 3, :] += 10.0
        outputs[3, 3, :] += 42.0

        self.assertEqual(
            self.task.predict(idxs, features, outputs),
            {0: "green", 1: "slow", 2: "slow"},
        )

    def test_score(self):
        pred = {0: "green", 1: "dog fox", 2: "slides"}
        f1, em = self.task.score(pred, self.examples, strict=False)
        self.assertAlmostEqual(f1, 5 / 9)
        self.assertAlmostEqual(em, 1 / 3)


class FunctionsTestCase(unittest.TestCase):
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

        overall = scores_to_overall(scores)
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
        metrics = scores_to_metrics(scores)
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
