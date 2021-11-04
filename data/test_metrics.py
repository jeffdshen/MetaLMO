import unittest
from .metrics import metric_accuracy, metric_f1, metric_multirc, metric_record


class MetricAccuracyTestCase(unittest.TestCase):
    def test_accurate(self):
        examples = [{"idx": 0, "label": "hello"}, {"idx": 2, "label": "world"}]
        pred = {0: "hello", 1: "!", 2: "world"}
        acc = metric_accuracy(pred, examples, strict=True)
        self.assertAlmostEqual(acc, 1.0)

    def test_inaccurate(self):
        examples = [{"idx": 0, "label": "hello"}, {"idx": 2, "label": "world"}]
        pred = {0: "hello", 2: "world!"}
        acc = metric_accuracy(pred, examples, strict=True)
        self.assertAlmostEqual(acc, 0.5)

    @unittest.expectedFailure
    def test_strict(self):
        examples = [{"idx": 0, "label": "hello"}, {"idx": 2, "label": "world"}]
        pred = {0: "hello"}
        acc = metric_accuracy(pred, examples, strict=True)
        self.assertAlmostEqual(acc, 1.0)

    def test_unstrict(self):
        examples = [{"idx": 0, "label": "hello"}, {"idx": 2, "label": "world"}]
        pred = {0: "hello"}
        acc = metric_accuracy(pred, examples, strict=False)
        self.assertAlmostEqual(acc, 1.0)


class MetricF1TestCase(unittest.TestCase):
    def test_accurate(self):
        examples = [{"idx": 0, "label": "a"}, {"idx": 2, "label": "b"}]
        pred = {0: "a", 1: "!", 2: "b"}
        acc = metric_f1(pred, examples, strict=True)
        self.assertAlmostEqual(acc, 1.0)

    def test_inaccurate(self):
        examples = [
            {"idx": 0, "label": "a"},
            {"idx": 1, "label": "c"},
            {"idx": 2, "label": "b"},
            {"idx": 3, "label": "c"},
        ]
        pred = {
            0: "a",
            1: "b",
            2: "b",
            3: "a",
            4: "c",
        }
        acc = metric_f1(pred, examples, strict=True)
        self.assertAlmostEqual(acc, (2 / 3 + 2 / 3 + 0) / 3)

    @unittest.expectedFailure
    def test_strict(self):
        examples = [
            {"idx": 0, "label": "a"},
            {"idx": 1, "label": "c"},
            {"idx": 2, "label": "b"},
            {"idx": 3, "label": "d"},
            {"idx": 4, "label": "b"},
            {"idx": 5, "label": "c"},
        ]
        pred = {
            0: "a",
            1: "b",
            2: "b",
            5: "a",
        }
        acc = metric_f1(pred, examples, strict=True)
        self.assertAlmostEqual(acc, (2 / 3 + 2 / 3 + 0) / 3)

    def test_unstrict(self):
        examples = [
            {"idx": 0, "label": "a"},
            {"idx": 1, "label": "c"},
            {"idx": 2, "label": "b"},
            {"idx": 3, "label": "d"},
            {"idx": 4, "label": "b"},
            {"idx": 5, "label": "c"},
        ]
        pred = {
            0: "a",
            1: "b",
            2: "b",
            5: "a",
            6: "c",
        }
        acc = metric_f1(pred, examples, strict=False)
        self.assertAlmostEqual(acc, (2 / 3 + 2 / 3 + 0) / 3)


class MetricRecordTestCase(unittest.TestCase):
    def test_accurate(self):
        examples = [
            {
                "qas": [
                    {"idx": 0, "answers": [{"text": "hello world"}]},
                    {"idx": 1, "answers": [{"text": "a b c"}]},
                ]
            },
            {
                "qas": [
                    {"idx": 2, "answers": [{"text": "d e"}]},
                ]
            },
        ]
        pred = {
            0: "hello world",
            1: "a b c",
            2: "d e",
            3: "foo bar",
        }
        f1, em = metric_record(pred, examples, strict=True)
        self.assertAlmostEqual(f1, 1.0)
        self.assertAlmostEqual(em, 1.0)

    def test_inaccurate(self):
        examples = [
            {
                "qas": [
                    {"idx": 0, "answers": [{"text": "hello world"}]},
                    {"idx": 1, "answers": [{"text": "a b c"}]},
                ]
            },
            {
                "qas": [
                    {"idx": 2, "answers": [{"text": "d e"}]},
                ]
            },
        ]
        pred = {
            0: "hello",
            1: "a b c",
            2: "d e f",
            3: "foo bar",
        }
        f1, em = metric_record(pred, examples, strict=True)
        self.assertAlmostEqual(f1, (2 / 3 + 1 + 4 / 5) / 3)
        self.assertAlmostEqual(em, 1 / 3)

    @unittest.expectedFailure
    def test_strict(self):
        examples = [
            {
                "qas": [
                    {"idx": 0, "answers": [{"text": "hello world"}]},
                    {"idx": 1, "answers": [{"text": "a b c"}]},
                ]
            },
            {
                "qas": [
                    {"idx": 2, "answers": [{"text": "foo bar"}]},
                    {"idx": 3, "answers": [{"text": "d e"}]},
                ]
            },
        ]
        pred = {
            0: "hello",
            1: "a b c",
            3: "d e f",
            4: "foo bar",
        }
        f1, em = metric_record(pred, examples, strict=True)
        self.assertAlmostEqual(f1, (2 / 3 + 1 + 4 / 5) / 3)
        self.assertAlmostEqual(em, 1 / 3)

    def test_unstrict(self):
        examples = [
            {
                "qas": [
                    {"idx": 0, "answers": [{"text": "hello world"}]},
                    {"idx": 1, "answers": [{"text": "a b c"}]},
                ]
            },
            {
                "qas": [
                    {"idx": 2, "answers": [{"text": "foo bar"}]},
                    {"idx": 3, "answers": [{"text": "d e"}]},
                ]
            },
        ]
        pred = {
            0: "hello",
            1: "a b c",
            3: "d e f",
            4: "foo bar",
        }
        f1, em = metric_record(pred, examples, strict=False)
        self.assertAlmostEqual(f1, (2 / 3 + 1 + 4 / 5) / 3)
        self.assertAlmostEqual(em, 1 / 3)


class MetricMultiRCTestCase(unittest.TestCase):
    def test_accurate(self):
        examples = [
            {
                "passage": {
                    "questions": [
                        {
                            "answers": [
                                {"idx": 0, "label": 1},
                                {"idx": 1, "label": 0},
                                {"idx": 2, "label": 1},
                            ]
                        },
                        {
                            "answers": [
                                {"idx": 3, "label": 0},
                                {"idx": 4, "label": 0},
                            ]
                        },
                    ]
                }
            },
            {
                "passage": {
                    "questions": [
                        {
                            "answers": [
                                {"idx": 5, "label": 1},
                                {"idx": 6, "label": 0},
                            ]
                        },
                    ]
                }
            },
        ]
        pred = {0: 1, 1: 0, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0}
        f1, em = metric_multirc(pred, examples, strict=True)
        self.assertAlmostEqual(f1, 1.0)
        self.assertAlmostEqual(em, 1.0)

    def test_inaccurate(self):
        examples = [
            {
                "passage": {
                    "questions": [
                        {
                            "answers": [
                                {"idx": 0, "label": 1},
                                {"idx": 1, "label": 0},
                                {"idx": 2, "label": 1},
                            ]
                        },
                        {
                            "answers": [
                                {"idx": 3, "label": 0},
                                {"idx": 4, "label": 0},
                            ]
                        },
                    ]
                }
            },
            {
                "passage": {
                    "questions": [
                        {
                            "answers": [
                                {"idx": 5, "label": 1},
                                {"idx": 6, "label": 0},
                            ]
                        },
                    ]
                }
            },
        ]
        pred = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 0}
        f1, em = metric_multirc(pred, examples, strict=True)
        self.assertAlmostEqual(f1, 4 / (4 + 2))
        self.assertAlmostEqual(em, 1 / 3)

    @unittest.expectedFailure
    def test_strict(self):
        examples = [
            {
                "passage": {
                    "questions": [
                        {
                            "answers": [
                                {"idx": 0, "label": 1},
                                {"idx": 1, "label": 0},
                                {"idx": 2, "label": 1},
                            ]
                        },
                        {
                            "answers": [
                                {"idx": 3, "label": 0},
                                {"idx": 4, "label": 0},
                                {"idx": 7, "label": 1},
                            ]
                        },
                    ]
                }
            },
            {
                "passage": {
                    "questions": [
                        {
                            "answers": [
                                {"idx": 5, "label": 1},
                                {"idx": 6, "label": 0},
                            ]
                        },
                        {
                            "answers": [
                                {"idx": 8, "label": 1},
                                {"idx": 9, "label": 0},
                            ]
                        },
                    ],
                }
            },
        ]
        pred = {
            0: 1,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 1,
            6: 1,
        }
        f1, em = metric_multirc(pred, examples, strict=True)
        self.assertAlmostEqual(f1, 4 / (4 + 2))
        self.assertAlmostEqual(em, 1 / 3)

    def test_unstrict(self):
        examples = [
            {
                "passage": {
                    "questions": [
                        {
                            "answers": [
                                {"idx": 0, "label": 1},
                                {"idx": 1, "label": 0},
                                {"idx": 2, "label": 1},
                            ]
                        },
                        {
                            "answers": [
                                {"idx": 3, "label": 0},
                                {"idx": 4, "label": 0},
                                {"idx": 7, "label": 1},
                            ]
                        },
                    ]
                }
            },
            {
                "passage": {
                    "questions": [
                        {
                            "answers": [
                                {"idx": 5, "label": 1},
                                {"idx": 6, "label": 0},
                            ]
                        },
                        {
                            "answers": [
                                {"idx": 8, "label": 1},
                                {"idx": 9, "label": 0},
                            ]
                        },
                    ],
                }
            },
        ]
        pred = {
            0: 1,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 1,
            6: 1,
        }
        f1, em = metric_multirc(pred, examples, strict=False)
        self.assertAlmostEqual(f1, 4 / (4 + 2))
        self.assertAlmostEqual(em, 1 / 3)


if __name__ == "__main__":
    unittest.main()
