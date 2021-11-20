import unittest

import torch

from .stats import (
    JoinTextFormatter,
    tensors_groupby_flatten,
    tensors_to_lists,
    StrTextFormatter,
)


class TensorsTestCase(unittest.TestCase):
    def test_tensors_to_lists(self):
        a = torch.arange(15).view(3, 5)
        b = torch.arange(15, 30).view(3, 5)
        c = torch.arange(30, 45).view(3, 5)

        x = tensors_to_lists([a, b, c])
        expected = [
            ([0, 1, 2, 3, 4], [15, 16, 17, 18, 19], [30, 31, 32, 33, 34]),
            ([5, 6, 7, 8, 9], [20, 21, 22, 23, 24], [35, 36, 37, 38, 39]),
            ([10, 11, 12, 13, 14], [25, 26, 27, 28, 29], [40, 41, 42, 43, 44]),
        ]
        self.assertEqual(x, expected)

    def test_tensors_groupby_flatten(self):
        self.maxDiff = None
        idxs = torch.tensor([0, 0, 1], dtype=torch.long)
        a = torch.arange(15).view(3, 5)
        b = torch.arange(15, 30).view(3, 5)
        c = torch.arange(30, 45).view(3, 5)

        x = tensors_groupby_flatten(idxs, [a, b, c])
        expected = {
            0: (
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            ),
            1: (
                [10, 11, 12, 13, 14],
                [25, 26, 27, 28, 29],
                [40, 41, 42, 43, 44],
            ),
        }
        self.assertEqual(x, expected)


class StrTextFormatterTestcase(unittest.TestCase):
    def test_format(self):
        formatter = StrTextFormatter(["a", "b", "c"])
        text = formatter([0, "d", {"e": "f"}])
        expected = "- **a:** 0\n- **b:** 'd'\n- **c:** {'e': 'f'}"
        self.assertEqual(text, expected)


class JoinTextFormatterTestcase(unittest.TestCase):
    def test_format(self):
        a = StrTextFormatter(["a", "b"])
        b = StrTextFormatter(["c"])
        formatter = JoinTextFormatter([a, b])
        text = formatter([0, "d", {"e": "f"}])
        expected = "- **a:** 0\n- **b:** 'd'\n- **c:** {'e': 'f'}"
        self.assertEqual(text, expected)
