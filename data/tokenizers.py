import pathlib

from tokenizers import ByteLevelBPETokenizer


class TextIterable:
    def __init__(self, dataset, split, column):
        self.dataset = dataset
        self.split = split
        self.column = column

    def __iter__(self):
        return map(lambda x: self.dataset[self.split][x][self.column], range(len(self)))

    def __len__(self):
        return len(self.dataset[self.split])


class BatchedTextIterable:
    def __init__(self, dataset, split, column, batch_size):
        self.dataset = dataset
        self.split = split
        self.column = column
        self.batch_size = batch_size

    def __iter__(self):
        return map(
            lambda x: self.dataset[self.split][x : x + self.batch_size][self.column],
            range(0, len(self), self.batch_size),
        )

    def __len__(self):
        return len(self.dataset[self.split])


def from_pretrained(path, **kwargs):
    vocab_file = str(pathlib.Path(path, "vocab.json"))
    merges_file = str(pathlib.Path(path, "merges.txt"))
    return ByteLevelBPETokenizer.from_file(vocab_file, merges_file, **kwargs)