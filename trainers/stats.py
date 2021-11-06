# Copyright (c) Jeffrey Shen

import random

class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def add(self, avg, count=1):
        self.count += count
        self.sum += avg * count
        self.avg = self.sum / self.count


class Visualizer:
    def __init__(self, keys, num_samples, func=None):
        self.keys = keys
        self.num_samples = num_samples
        self.func = func

    def sample(self, items):
        items = random.sample(items, k=max(0, min(self.num_samples, len(items))))
        if self.func is not None:
            items = [tuple(self.func(x) for x in sample) for sample in items]
        keys = ["- **{}:** {{}}".format(x) for x in self.keys]
        items = ["\n".join(keys[i].format(x) for i, x in enumerate(sample)) for sample in items]
        return items

    