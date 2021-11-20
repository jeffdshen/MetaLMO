# Copyright (c) Jeffrey Shen

import collections
import logging
import os
import random
from torch.utils.data.dataset import T

import tqdm


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


class TensorboardScalars:
    def __init__(self, tbx, group, tags):
        super().__init__()
        self.tbx = tbx
        self.group = group
        self.tags = tags

    def __call__(self, info, step):
        for tag in self.tags:
            value = info[tag]
            self.tbx.add_scalar("{}/{}".format(self.group, tag), value, step)


class TensorboardWeights:
    def __init__(self, tbx, group):
        super().__init__()
        self.tbx = tbx
        self.group = group

    def __call__(self, model, step):
        for tags, params in model.named_parameters():
            self.tbx.add_histogram("{}/{}".format(self.group, tags), params.data, step)


class TensorboardText:
    def __init__(self, tbx, group, tag, formatter, max_samples):
        super().__init__()
        self.tbx = tbx
        self.group = group
        self.tag = tag
        self.max_samples = max_samples
        self.samples = collections.deque()
        self.formatter = formatter

    def add_all(self, items):
        for item in items:
            self.add(item)

    def add(self, item):
        self.samples.append(item)
        if len(self.samples) > self.max_samples:
            self.samples.popleft()

    def clear(self):
        self.samples.clear()

    def __call__(self, step):
        for i, sample in enumerate(self.samples):
            self.tbx.add_text(
                "{}/{}_{}_of_{}".format(self.group, self.tag, i + 1, self.max_samples),
                self.formatter(sample),
                step,
            )


def tensors_to_lists(tensors):
    samples = [tensor.tolist() for tensor in tensors]
    samples = list(zip(*samples))
    return samples


def tensors_groupby_flatten(idxs, tensors):
    unique_idxs, counts = idxs.unique_consecutive(dim=0, return_counts=True)
    tensors = [[t.flatten().tolist() for t in tensor.split(counts.tolist())] for tensor in tensors]
    return {
        idx.item(): items
        for idx, items in zip(unique_idxs, list(zip(*tensors)))
    }


class TokenizedTextFormatter:
    def __init__(self, tokenizer, keys, skip_special_tokens=False):
        self.tokenizer = tokenizer
        self.keys = keys
        self.skip_special_tokens = skip_special_tokens

    def __call__(self, sample):
        sample = [self.tokenizer.decode(x, self.skip_special_tokens) for x in sample]
        keys = ["- **{}:** {{}}".format(x) for x in self.keys]
        text = "\n".join(k.format(repr(x)) for k, x in zip(keys, sample))
        return text

    def __len__(self):
        return len(self.keys)
    
class StrTextFormatter:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        keys = ["- **{}:** {{}}".format(x) for x in self.keys]
        text = "\n".join(k.format(repr(x)) for k, x in zip(keys, sample))
        return text

    def __len__(self):
        return len(self.keys)


class JoinTextFormatter:
    def __init__(self, formatters):
        self.formatters = formatters

    def __call__(self, sample):
        i = 0
        texts = []
        for formatter in self.formatters:
            texts.append(formatter(sample[i:i + len(formatter)]))
            i += len(formatter)
        return "\n".join(texts)

    def __len__(self):
        return sum(len(formatter) for formatter in self.formatters)


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
        items = [
            "\n".join(keys[i].format(x) for i, x in enumerate(sample))
            for sample in items
        ]
        return items


def get_logger(log_dir, name):
    class TqdmStreamHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

    logger = logging.getLogger(name)
    # Check if already set
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(log_dir, "log.txt")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y.%m.%d %H:%M:%S")
    )
    logger.addHandler(file_handler)

    console_handler = TqdmStreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y.%m.%d %H:%M:%S")
    )
    logger.addHandler(console_handler)

    return logger
