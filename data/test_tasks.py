import unittest

import torch
import torch.nn.functional as F

from .tasks import get_mlm_task
from .tokenizers import get_fake_tokenizer
import data.config


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
        self.task = data.config.nlp.get_tasks(tokenizer)["BoolQ"]
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
        t = self.task.task_id
        idxs = [0, 0, 1, 1, 2]
        features = [[t, 2], [2, 2], [t, 2], [t, 0], [1, 0]]
        outputs = torch.full((5, 2, 512), 0.0, dtype=torch.float32)

        idxs = torch.tensor(idxs, dtype=torch.long)
        features = torch.tensor(features, dtype=torch.long)
        outputs[:, :, t] = 1.0
        outputs[:, :, 2] = 1.0

        # test average
        outputs[0, 0, t] += 1.0
        outputs[1, 1, 2] += 2.0
        outputs[2, 0, 2] += 1.1
        outputs[3, 0, t] += 1.0
        outputs[4, 0, t] += 1.0

        # padding and non-classes don't matter
        outputs[:, :, t - 1] = 10.0
        outputs[4, 1, t] = 10.0
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


class MLMTestCase(unittest.TestCase):
    def setUp(self):
        self.task = get_mlm_task(tokenizer, 0.25, 0.1, 0.1)
        self.examples = [tokenizer.encode(a, b).ids for a, b in examples]
        self.examples = [
            (torch.tensor([i], dtype=torch.long), torch.tensor(e, dtype=torch.long))
            for i, e in enumerate(self.examples)
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
                self.assertEqual(idxs.size(0), 1)

                labels = labels.squeeze(0)
                features = features.squeeze(0)
                self.assertTorchScalarEqual(features[0], self.task.task_id)
                self.assertAlmostEqual(
                    (labels != 0).sum().item(),
                    self.task.mask_prob * labels.size(0),
                    delta=1.0,
                )

                mask = labels == 0
                labels[mask] = features[mask]
                labels[0] = tokenizer.token_to_id("[CLS0]")

                self.assertTorchEqual(idxs, example[0])
                self.assertTorchEqual(labels, example[1])

    def test_predict(self):
        t = self.task.task_id
        idxs = [0, 1, 2, 3, 4]
        features = [[t, 2], [2, 2], [t, 2], [t, 0], [1, 0]]
        outputs = torch.full((5, 2, 512), 0.0, dtype=torch.float32)

        idxs = torch.tensor(idxs, dtype=torch.long)
        features = torch.tensor(features, dtype=torch.long)
        outputs[:, :, t] = 1.1
        outputs[:, :, 2] = 1.0

        # test argmax
        outputs[0, 0, 50] += 2.0
        outputs[1, 1, 2] += 1.0
        outputs[2, 0, 51] += 1.2
        outputs[2, 1, 52] += 1.2
        outputs[3, 0, 53] += 1.3
        outputs[4, 0, 2] += 1.0

        # padding doesn't matter
        outputs[4, 1, 49] = 10.0
        outputs[3, 1, 48] = 10.0

        # constant shift doesn't matter
        outputs[1, 0, :] += 100.0

        self.assertEqual(
            self.task.predict(idxs, features, outputs),
            {0: [50, t], 1: [t, 2], 2: [51, 52], 3: [53, 0], 4: [2, 0]},
        )

        # answer doesn't change even if we do log_softmax
        outputs = F.log_softmax(outputs, dim=-1)
        self.assertEqual(
            self.task.predict(idxs, features, outputs),
            {0: [50, t], 1: [t, 2], 2: [51, 52], 3: [53, 0], 4: [2, 0]},
        )

    def test_score(self):
        pred = {0: [0, 50, 52, 0], 1: [4, 51, 51, 0], 3: [0, 0, 53, 0]}
        labels = [[4, 50, 51, 2], [4, 51, 52, 2], [1, 2, 3, 4], [4, 51, 53, 2]]
        labels = [torch.tensor(label, dtype=torch.long) for label in labels]
        examples = list(enumerate(labels))

        score = self.task.score(pred, examples, strict=False)
        self.assertAlmostEqual(score, (1 / 2 + 2 / 3 + 1) / 3)
