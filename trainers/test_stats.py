import unittest

import torch

from .stats import tensors_to_lists


class TensorsToListsTestCase(unittest.TestCase):
    def test_tensors_to_lists(self):
        a = torch.arange(15).view(3, 5)
        b = torch.arange(15, 30).view(3, 5)
        c = torch.arange(30, 45).view(3, 5)

        x = tensors_to_lists([a, b, c])
        expected = (
            ([0, 1, 2, 3, 4], [15, 16, 17, 18, 19], [30, 31, 32, 33, 34]),
            ([5, 6, 7, 8, 9], [20, 21, 22, 23, 24], [35, 36, 37, 38, 39]),
            ([10, 11, 12, 13, 14], [25, 26, 27, 28, 29], [40, 41, 42, 43, 44]),
        )
        self.assertEqual(x, expected)