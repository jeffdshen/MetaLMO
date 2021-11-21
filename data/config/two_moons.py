import numpy as np
import sklearn.datasets

from data.datasets import FlatDataset, FlatPretrainDataset
from data.tasks import WhichMoonTask


def gen_moons(n_samples):
    x, labels = sklearn.datasets.make_moons(
        n_samples=n_samples, shuffle=True, noise=0.10, random_state=42
    )
    x, y = x[:, 0], x[:, 1]
    x = np.interp(x, (-1.5, 2.5), (0, 99.99)).astype(int)
    x = np.clip(x, 0, 99)
    y = np.interp(y, (-1.0, 1.5), (0, 99.99)).astype(int)
    y = np.clip(y, 0, 99)
    return x, y, labels


def gen_sequences(x, y, labels, start, end, seq_size):
    if (end - start) % seq_size != 0:
        raise ValueError("size must be a multiple of seq_size")
    seqs = []
    for i in range(start, end, seq_size):
        xy = np.stack((x[i : i + seq_size], y[i : i + seq_size]), axis=-1)
        xy = np.array(sorted(xy.tolist()))
        seq = " ".join(str(p) for p in xy.flatten().tolist())
        seqs.append(seq)
    return seqs


def gen_which_moon(x, y, labels, start, end):
    examples = []
    for i in range(start, end):
        seq = " ".join([str(x[i]), str(y[i])])
        examples.append({"idx": (i - start), "question": seq, "label": labels[i]})
    return examples


def get_raw_data():
    x, y, labels = gen_moons(1_000_000)
    all_x, all_y = np.meshgrid(np.arange(0, 100), np.arange(0, 100))
    all_x, all_y = all_x.flatten(), all_y.flatten()
    data = {}
    data["TWO_MOONS"] = gen_sequences(x, y, labels, 0, 512000, 16)

    data["Which_MOON"] = {}
    data["Which_MOON"]["train"] = gen_which_moon(x, y, labels, 512010, 512020)
    data["Which_MOON"]["val"] = gen_which_moon(x, y, labels, 512030, 512040)
    data["Which_MOON"]["test"] = gen_which_moon(
        all_x, all_y, np.zeros_like(all_x), 0, 10000
    )

    return data


def get_pretrain_dataset(data, tokenizer):
    return FlatPretrainDataset(data["TWO_MOONS"], tokenizer)


def get_raw_task_datasets(data):
    def to_dataset(splits, dataset_class):
        return {k: dataset_class(v) for k, v in splits.items()}

    def to_moon_dataset(name, dataset_class):
        return to_dataset(data[name], dataset_class)

    return {
        "Which_MOON": to_moon_dataset("Which_MOON", FlatDataset),
    }


def get_tasks(tokenizer):
    task_ids = {
        "Which_MOON": 2,
    }
    task_ids = {
        k: tokenizer.token_to_id("[CLS{}]".format(v)) for k, v in task_ids.items()
    }
    sep_id = tokenizer.token_to_id("[SEP]")
    mask_id = tokenizer.token_to_id("[MASK]")
    labels = {"Which_MOON": [(0, sep_id), (1, task_ids["Which_MOON"])]}
    return {
        "Which_MOON": WhichMoonTask(
            task_ids["Which_MOON"], tokenizer, labels["Which_MOON"]
        )
    }


class Scorer:
    dataset_names = ["Which_MOON"]

    @staticmethod
    def scores_to_overall(scores):
        moons = ["Which_MOON"]
        moons_score = np.mean(
            [np.mean(scores[name]) for name in moons if name in scores]
        )
        return {"Overall": moons_score, "TWO_MOONS": moons_score}

    @staticmethod
    def scores_to_metrics(scores):
        metrics = scores.copy()
        return metrics

    @staticmethod
    def get_overall_names():
        return ["Overall", "TWO_MOONS"]

    @staticmethod
    def get_maximize_metrics():
        return {
            "Overall": True,
            "TWO_MOONS": True,
        }

    @classmethod
    def get_dataset_names(cls):
        return cls.dataset_names

    @staticmethod
    def get_metric_names(score_names=dataset_names):
        metrics = []
        metrics_map = {}
        for name in score_names:
            if name in metrics_map:
                metrics += metrics_map[name]
            else:
                metrics.append(name)

        return metrics
