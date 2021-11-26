import unittest

import torch

from .transformer import SoftLearnedTokenEmbedding


class SoftLearnedTokenEmbeddingTestCase(unittest.TestCase):
    def assertTorchAlmostEqual(self, a, b):
        c = (a - b).detach().flatten().tolist()
        for i, x in enumerate(c):
            self.assertAlmostEqual(x, 0.0, msg=f"Not equal at {i}: {a}, {b}", places=6)

    def test_forward(self):
        e = SoftLearnedTokenEmbedding(15, 5, padding_idx=1)
        x = torch.arange(15).view(5, 3)
        y = e(x)
        self.assertTorchAlmostEqual(y.flatten(end_dim=1), e.embed.weight.data)
        self.assertTorchAlmostEqual(y[0, 1], 0.0)
        y.sum().backward()
        expected_grad = torch.full((15, 5), 1.0)
        expected_grad[1] = 0.0
        self.assertTorchAlmostEqual(e.embed.weight.grad, expected_grad)

        e.embed.weight.data[1][0] = 1.0
        x = torch.full((1, 15), 0.0)
        x[0, 1] = 1.0
        y = e(x)
        self.assertEqual(list(y.size()), [1, 5])
        self.assertTorchAlmostEqual(y, torch.tensor([1.0, 0, 0, 0, 0]))
        y.sum().backward()
        self.assertTorchAlmostEqual(e.embed.weight.grad, expected_grad)

        x = (torch.arange(30) * 0.1 + 0.1).view(2, 15)
        y = e(x)
        self.assertEqual(list(y.size()), [2, 5])
        y.sum().backward()
        for i in range(15):
            expected_grad[i] += 1.7 + i * 0.2
        expected_grad[1] = 0.0
        self.assertTorchAlmostEqual(e.embed.weight.grad, expected_grad)
