import numbers
import numpy as np
import sklearn.datasets

from data.datasets import FlatDataset, FlatPretrainDataset, MiniDataset
from data.tasks import WhichMoonTask
from .util import add_all_to_overall, add_mean_to_overall, add_suffix


def to_str(num, base, length):
    s = np.base_repr(num, base=base, padding=length)
    return s[-length:]


def gen_moons(n_samples):
    x, labels = sklearn.datasets.make_moons(
        n_samples=n_samples, shuffle=True, noise=0.10, random_state=42
    )
    x, y = x[:, 0], x[:, 1]
    x = np.interp(x, (-1.5, 2.5), (0, 63.99)).astype(int)
    x = np.clip(x, 0, 63)
    y = np.interp(y, (-1.0, 1.5), (0, 63.99)).astype(int)
    y = np.clip(y, 0, 63)
    return x, y, labels


def gen_sequences(x, y, labels, start, end, seq_size):
    if (end - start) % seq_size != 0:
        raise ValueError("size must be a multiple of seq_size")
    seqs = []
    for i in range(start, end, seq_size):
        xy = np.stack((x[i : i + seq_size], y[i : i + seq_size]), axis=-1)
        xy = np.array(sorted(xy.tolist()))
        seq = " ".join(to_str(p, base=4, length=3) for p in xy.flatten().tolist())
        seqs.append(seq)
    return seqs


def gen_which_moon(x, y, labels, start, end):
    examples = []
    for i in range(start, end):
        seq = " ".join([to_str(p, base=4, length=3) for p in [x[i], y[i]]])
        examples.append({"idx": (i - start), "question": seq, "label": labels[i]})
    return examples


def get_raw_data():
    x, y, labels = gen_moons(1_000_000)
    all_x, all_y = np.meshgrid(np.arange(0, 100), np.arange(0, 100))
    all_x, all_y = all_x.flatten(), all_y.flatten()
    data = {}
    data["TWO_MOONS"] = {}
    data["TWO_MOONS"]["train"] = gen_sequences(x, y, labels, 0, 512_000, 16)
    data["TWO_MOONS"]["val"] = gen_sequences(x, y, labels, 600_000, 616_000, 16)

    data["Which_MOON"] = {}
    data["Which_MOON"]["train"] = gen_which_moon(x, y, labels, 512_010, 512_020)
    data["Which_MOON"]["val"] = gen_which_moon(x, y, labels, 512_030, 512_040)
    # TODO:NOTE: Don't name this test so that scores get reported back
    data["Which_MOON"]["test_score"] = gen_which_moon(x, y, labels, 512_100, 512_200)
    data["Which_MOON"]["vis"] = gen_which_moon(
        all_x, all_y, np.zeros_like(all_x), 0, 10_000
    )

    return data


def get_pretrain_datasets(data, tokenizer, mini_val_size):
    val_dataset = FlatPretrainDataset(data["TWO_MOONS"]["val"], tokenizer)
    mini_val_dataset = MiniDataset(val_dataset, mini_val_size)
    return {
        "train": FlatPretrainDataset(data["TWO_MOONS"]["train"], tokenizer),
        "val": val_dataset,
        "mini_val": mini_val_dataset,
    }


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
    all_names = dataset_names + ["MLM"] + add_suffix(dataset_names + ["MLM"], "_loss")
    metrics_map = {}

    @staticmethod
    def scale_scores(scores, multiplier=100):
        for k in scores:
            if isinstance(scores[k], numbers.Number):
                scores[k] *= 100
            else:
                scores[k] = tuple(v * multiplier for v in scores[k])

    @classmethod
    def scores_to_overall(cls, scores):
        overall = {}
        add_mean_to_overall(overall, scores, cls.dataset_names, "TWO_MOONS")
        add_mean_to_overall(
            overall, scores, add_suffix(cls.dataset_names, "_loss"), "TWO_MOONS_loss"
        )
        add_all_to_overall(overall, scores, ["MLM", "MLM_loss"])
        add_mean_to_overall(overall, overall, ["MLM", "TWO_MOONS"], "Overall")
        add_mean_to_overall(overall, overall, ["MLM_loss", "TWO_MOONS_loss"], "loss")
        return overall

    @classmethod
    def scores_to_metrics(cls, scores):
        metrics = scores.copy()
        for metric in cls.metrics_map:
            if metric in metrics:
                values = metrics.pop(metric)
                for k, v in zip(cls.metrics_map[metric], values):
                    metrics[k] = v

        return metrics

    @classmethod
    def get_overall_names(cls, score_names=all_names):
        overall = ["Overall", "loss"]
        if any(name in score_names for name in cls.dataset_names):
            overall.append("TWO_MOONS")

        if any(name in score_names for name in add_suffix(cls.dataset_names, "_loss")):
            overall.append("TWO_MOONS_loss")

        for metric in ["MLM", "MLM_loss"]:
            if metric in score_names:
                overall.append(metric)

        return overall

    @staticmethod
    def get_maximize_metrics():
        return {
            "Overall": True,
            "TWO_MOONS": True,
            "TWO_MOONS_loss": False,
            "MLM": True,
            "MLM_loss": False,
            "loss": False,
        }

    @classmethod
    def get_dataset_names(cls):
        return cls.dataset_names

    @classmethod
    def get_metric_names(cls, score_names=dataset_names):
        metrics = []
        for name in score_names:
            if name in cls.metrics_map:
                metrics += cls.metrics_map[name]
            else:
                metrics.append(name)

        return metrics
