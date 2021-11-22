import copy
import json
import pathlib
import random
from itertools import groupby

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence


class FlatDataset(Dataset):
    def __init__(self, examples):
        super().__init__()
        self.examples = examples
        self.index = list(range(len(examples)))

    def shuffle(self, seed):
        self.index = self.index.copy()
        random.Random(seed).shuffle(self.index)

    def __getitem__(self, idx):
        i = self.index[idx]
        return self.examples[i]

    def __len__(self):
        return len(self.index)

    def as_eval_dict(self):
        return self.examples


class MultiRCDataset(Dataset):
    def __init__(self, examples):
        super().__init__()
        self.examples = examples
        self.index = MultiRCDataset._index(examples)

    def shuffle(self, seed):
        groups = [list(g) for _, g in groupby(self.index, lambda i: (i[0], i[1]))]
        random.Random(seed).shuffle(groups)
        self.index = [x for group in groups for x in group]

    @staticmethod
    def _index(examples):
        index_map = {}
        for i, example in enumerate(examples):
            passage = example["passage"]
            questions = passage["questions"]
            for q, question in enumerate(questions):
                answers = question["answers"]
                for a, answer in enumerate(answers):
                    idx = answer["idx"]
                    if idx in index_map:
                        raise ValueError("Duplicate index in MultiRC: {}".format(idx))
                    index_map[idx] = (i, q, a)

        index = []
        for i in range(len(index_map)):
            if i not in index_map:
                raise ValueError("Missing index in MultiRC: {}".format(i))
            index.append(index_map[i])

        return index

    def __getitem__(self, idx):
        i, q, a = self.index[idx]
        example = self.examples[i].copy()
        example["passage"] = example["passage"].copy()
        passage = example["passage"]
        passage["questions"] = passage["questions"][q]
        passage["questions"] = passage["questions"].copy()
        passage["questions"]["answers"] = passage["questions"]["answers"][a]
        return example

    def __len__(self):
        return len(self.index)

    def as_eval_dict(self):
        return self.examples


class ReCoRDDataset(Dataset):
    def __init__(self, examples):
        super().__init__()
        self.examples = examples
        self.index = ReCoRDDataset._index(examples)

    @staticmethod
    def _index(examples):
        index_map = {}
        for i, example in enumerate(examples):
            qas = example["qas"]
            for q, qa in enumerate(qas):
                idx = qa["idx"]
                if idx in index_map:
                    raise ValueError("Duplicate index in ReCoRD: {}".format(idx))
                index_map[idx] = (i, q)

        index = []
        for i in range(len(index_map)):
            if i not in index_map:
                raise ValueError("Missing index in ReCoRD: {}".format(i))
            index.append(index_map[i])

        return index

    def shuffle(self, seed):
        self.index = self.index.copy()
        random.Random(seed).shuffle(self.index)

    def __getitem__(self, idx):
        i, q = self.index[idx]
        example = self.examples[i].copy()
        example["qas"] = example["qas"][q]
        return example

    def __len__(self):
        return len(self.index)

    def as_eval_dict(self):
        return self.examples


class MiniDataset(Dataset):
    def __init__(self, dataset, length):
        self.dataset = dataset
        self.length = length

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return min(self.length, len(self.dataset))

    def as_eval_dict(self):
        return self.dataset.as_eval_dict()


class TaskDataset(Dataset):
    def __init__(self, dataset, task, window, strict):
        super().__init__()
        self.dataset = dataset
        self.task = task
        self.window = window
        self.strict = strict

    def __getitem__(self, idx):
        return self.task.encode(self.dataset[idx], self.window)

    def __len__(self):
        return len(self.dataset)

    def predict(self, idxs, inputs, outputs):
        return self.task.predict(idxs, inputs, outputs)

    def score(self, preds):
        return self.task.score(preds, self.dataset.as_eval_dict(), self.strict)


class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.length = sum(self.lengths)

    def __len__(self):
        return self.length

    def sizes(self):
        return self.lengths

    def __getitem__(self, idx):
        d, i = idx
        return self.datasets[d][i]


class WikiDataset(Dataset):
    def __init__(self, dataset, split, column, tokenizer, cached_sizes=None):
        self.dataset = dataset
        self.split = split
        self.column = column
        self.tokenizer = tokenizer
        if cached_sizes is None:
            self.sizes = WikiDataset._compute_sizes(dataset, split, column)
        else:
            self.sizes = cached_sizes

    @staticmethod
    def _compute_sizes(dataset, split, column):
        sizes = np.full(len(dataset[split]), 0, dtype=np.int64)
        total = 0
        for i in tqdm(range(len(dataset[split]))):
            size = len(dataset[split][i][column])
            total += size
            sizes[i] = total
        return sizes

    def __getitem__(self, idx):
        i = np.searchsorted(self.sizes, idx, side="right").item()
        j = idx
        if i > 0:
            j -= self.sizes[i - 1]
        encoding = self.tokenizer.encode(self.dataset[self.split][i][self.column])
        encodings = [encoding] + encoding.overflowing
        for e in encodings:
            for start, end in e.offsets:
                if j >= start and j < end:
                    idx_tensor = torch.tensor([idx], dtype=torch.long)
                    return idx_tensor, torch.tensor(e.ids, dtype=torch.long)

        assert False

    def __len__(self):
        return self.sizes[-1]

    def as_eval_dict(self):
        return self


class MiniWikiDataset(Dataset):
    def __init__(self, dataset, split, column, tokenizer):
        self.dataset = dataset
        self.split = split
        self.column = column
        self.tokenizer = tokenizer
        self.sizes = MiniWikiDataset._compute_sizes(dataset, split, column, tokenizer)

    @staticmethod
    def _compute_sizes(dataset, split, column, tokenizer):
        sizes = np.full(len(dataset[split]), 0, dtype=np.int64)
        total = 0
        for i in tqdm(range(len(dataset[split]))):
            encoding = tokenizer.encode(dataset[split][i][column])
            size = 1 + len(encoding.overflowing)
            total += size
            sizes[i] = total
        return sizes

    def __getitem__(self, idx):
        i = np.searchsorted(self.sizes, idx, side="right").item()
        j = idx
        if i > 0:
            j -= self.sizes[i - 1]
        encoding = self.tokenizer.encode(self.dataset[self.split][i][self.column])
        encodings = [encoding] + encoding.overflowing
        idx_tensor = torch.tensor([idx], dtype=torch.long)
        return idx_tensor, torch.tensor(encodings[j].ids, dtype=torch.long)

    def __len__(self):
        return self.sizes[-1]

    def as_eval_dict(self):
        return self


class FlatPretrainDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        idx_tensor = torch.tensor([idx], dtype=torch.long)
        encoding = self.tokenizer.encode(self.dataset[idx])
        return idx_tensor, torch.tensor(encoding.ids, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def as_eval_dict(self):
        return self


class PretrainTaskDataset(Dataset):
    def __init__(self, dataset, task, window, strict):
        super().__init__()
        self.dataset = dataset
        self.task = task
        self.window = window
        self.strict = strict

    def __getitem__(self, idx):
        idxs, features, labels = self.task.encode(self.dataset[idx], self.window)
        return self.dataset[idx] + (idxs, features.squeeze(0), labels.squeeze(0))

    def __len__(self):
        return len(self.dataset)

    def predict(self, idxs, inputs, outputs):
        return self.task.predict(idxs, inputs, outputs)

    def score(self, preds):
        return self.task.score(preds, self.dataset.as_eval_dict(), self.strict)


class MetaDataset(Dataset):
    def __init__(self, pretrain_dataset, multi_dataset):
        self.pretrain_dataset = pretrain_dataset
        self.multi_dataset = multi_dataset

    def __getitem__(self, idx):
        pretrain_idx, multi_idx_s, multi_idx_q = idx
        return (
            self.pretrain_dataset[pretrain_idx],
            self.multi_dataset[multi_idx_s],
            self.multi_dataset[multi_idx_q],
        )

    def sizes(self):
        return len(self.pretrain_dataset), self.multi_dataset.sizes()

    def __len__(self):
        return (
            len(self.pretrain_dataset)
            * len(self.multi_dataset)
            * len(self.multi_dataset)
        )


class TaskCollater:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        idxs, features, labels = zip(*examples)
        idxs = torch.cat(idxs, dim=0)
        features = [t for f in features for t in f]
        features = pad_sequence(features, batch_first=True, padding_value=self.pad_id)
        labels = [t for l in labels for t in l]
        labels = pad_sequence(labels, batch_first=True, padding_value=self.pad_id)
        return idxs, features, labels


class SequenceCollater:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        return pad_sequence(examples, batch_first=True, padding_value=self.pad_id)


class TupleSequenceCollater:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        sequences = list(zip(*examples))
        return tuple(
            pad_sequence(sequence, batch_first=True, padding_value=self.pad_id)
            for sequence in sequences
        )


class MetaCollater:
    def __init__(self, pad_id):
        self.task_collater = TaskCollater(pad_id)
        self.sequence_collater = TupleSequenceCollater(pad_id)

    def __call__(self, examples):
        meta_examples, support_examples, query_examples = zip(*examples)
        return (
            self.sequence_collater(meta_examples),
            self.task_collater(support_examples),
            self.task_collater(query_examples),
        )


class MetaSampler(Sampler):
    def __init__(self, dataset: MetaDataset, num_samples: int, samples_per_task: int):
        self.dataset = dataset
        self.num_samples = num_samples
        self.samples_per_task = samples_per_task

    def __iter__(self):
        items = []
        meta_size, multi_sizes = self.dataset.sizes()
        for i in range(self.num_samples):
            meta_idx = random.randrange(meta_size)
            if i % self.samples_per_task == 0:
                task_idx = random.randrange(len(multi_sizes))
            example_idx_s = random.randrange(multi_sizes[task_idx])
            example_idx_q = random.randrange(multi_sizes[task_idx] - 1)
            if example_idx_q >= example_idx_s:
                example_idx_q += 1
            items.append(
                (meta_idx, (task_idx, example_idx_s), (task_idx, example_idx_q))
            )
        for item in items:
            yield item

    def __len__(self):
        return self.num_samples


def read_jsonl(path):
    with open(path) as file:
        data = [json.loads(line) for line in file]
    return data


def read_splits(path, splits, ext):
    return {split: read_jsonl(pathlib.Path(path, split + ext)) for split in splits}


def get_task_datasets(raw_datasets, tasks, mini_val_size):
    task_datasets = {}
    for name, splits in raw_datasets.items():
        task = tasks[name]
        task_datasets[name] = {}
        for split, dataset in splits.items():
            window = "random" if split == "train" else "all"
            strict = split != "train"
            task_dataset = TaskDataset(dataset, task, window=window, strict=strict)
            task_datasets[name][split] = task_dataset

        for split in ["val"]:
            dataset = splits[split]
            dataset = copy.copy(dataset)
            dataset.shuffle(seed=42)
            dataset = MiniDataset(dataset, mini_val_size)
            dataset = TaskDataset(dataset, task, window="all", strict=False)
            task_datasets[name]["mini_val"] = dataset

    return task_datasets


def get_meta_dataset(pretrain_datasets, task_datasets, split):
    multi_dataset = MultiTaskDataset(
        [splits[split] for splits in task_datasets.values()]
    )
    return MetaDataset(pretrain_datasets[split], multi_dataset)
