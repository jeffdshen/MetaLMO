import unittest

import random
import torch
import numpy as np

from .state import TrainerState, SimpleState, KeepSimpleState, RandomState


class SimpleStateTestCase(unittest.TestCase):
    def test_init(self):
        x = SimpleState(a=10, b=100)
        self.assertEqual(x.a, 10)
        self.assertEqual(x.b, 100)

    def test_state_dict(self):
        x = SimpleState(a=10, b=100)
        self.assertEqual(x.state_dict(), {"a": 10, "b": 100})

    def test_modify(self):
        x = SimpleState(a=10, b=100)
        x.a = 11
        x.c = 99
        self.assertEqual(x.state_dict(), {"a": 11, "b": 100, "c": 99})

    def test_load_state_dict(self):
        x = SimpleState(a=10, b=100)
        state = {"a": 11, "c": 99}
        x.load_state_dict(state)
        self.assertEqual(x.state_dict(), {"a": 11, "c": 99})

    def test_init_from(self):
        x = SimpleState(a=10, b=100)
        y = SimpleState.init_from(x)
        self.assertEqual(y.state_dict(), {"a": 10, "b": 100})


class KeepSimpleStateTestCase(unittest.TestCase):
    def test_init(self):
        x = KeepSimpleState(["a"], a=10, b=100)
        self.assertEqual(x.a, 10)
        self.assertEqual(x.b, 100)

    def test_state_dict(self):
        x = KeepSimpleState(["a"], a=10, b=100)
        y = x.state_dict().copy()
        y.update({"a": 10, "b": 100})
        self.assertEqual(x.state_dict(), y)

    def test_modify(self):
        x = KeepSimpleState(["a"], a=10, b=100)
        x.a = 11
        x.c = 99
        y = x.state_dict().copy()
        y.update({"a": 11, "b": 100, "c": 99})
        self.assertEqual(x.state_dict(), y)

    def test_load_state_dict(self):
        x = KeepSimpleState(["a", "b"], a=10, b=100, c=1000)
        state = {"a": 11, "c": 99}
        x.load_state_dict(state)
        self.assertEqual(x.a, 10)
        self.assertEqual(x.b, 100)
        self.assertEqual(x.c, 99)

    def test_init_from(self):
        x = KeepSimpleState(["a"], a=10, b=100)
        y = KeepSimpleState.init_from(x, ["a", "b"])
        y.load_state_dict({"a": 9, "b": 99})
        self.assertEqual(y.a, 10)
        self.assertEqual(y.b, 100)


class RandomStateTestCase(unittest.TestCase):
    def test_init(self):
        _ = RandomState(seed=42)
        a = random.randint(0, 100)
        b = torch.randint(0, 100, size=(1,)).item()
        c = np.random.randint(0, 100)
        _ = RandomState(seed=42)
        self.assertEqual(random.randint(0, 100), a)
        self.assertEqual(torch.randint(0, 100, size=(1,)).item(), b)
        self.assertEqual(np.random.randint(0, 100), c)

    def test_state_dict(self):
        x = RandomState(seed=42)
        a_old = random.randint(0, 100)
        b_old = torch.randint(0, 100, size=(1,)).item()
        c_old = np.random.randint(0, 100)
        state = x.state_dict()
        a = random.randint(0, 100)
        b = torch.randint(0, 100, size=(1,)).item()
        c = np.random.randint(0, 100)
        self.assertNotEqual(a, a_old)
        self.assertNotEqual(b, b_old)
        self.assertNotEqual(c, c_old)
        x.load_state_dict(state)
        self.assertEqual(random.randint(0, 100), a)
        self.assertEqual(torch.randint(0, 100, size=(1,)).item(), b)
        self.assertEqual(np.random.randint(0, 100), c)


class TrainerStateTestCase(unittest.TestCase):
    def test_reload(self):
        x = TrainerState()
        self.assertFalse(x.is_reloading())
        obj1 = SimpleState(a="hello", b="world")
        x.track_object("obj1", obj1)
        obj2 = SimpleState(a="foo", c="bar")
        x.track_object("obj2", obj2)

        obj1.c = "!"
        obj2.d = "baz"
        state = x.state_dict()

        x = TrainerState()
        x.load_state_dict(state)
        self.assertTrue(x.is_reloading())
        obj1 = SimpleState()
        x.track_object("obj1", obj1)
        obj2 = SimpleState()
        x.track_object("obj2", obj2)
        self.assertEqual(obj1.state_dict(), {"a": "hello", "b": "world", "c": "!"})
        self.assertEqual(obj2.state_dict(), {"a": "foo", "c": "bar", "d": "baz"})
        self.assertFalse(x.is_reloading())
