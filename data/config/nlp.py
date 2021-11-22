import pathlib
import numbers

import numpy as np
from data.datasets import (
    FlatDataset,
    MiniWikiDataset,
    MultiRCDataset,
    ReCoRDDataset,
    WikiDataset,
    read_splits,
)
from datasets import load_dataset

from data.tasks import (
    BoolQTask,
    RTETask,
    ReCoRDTask,
    CBTask,
    COPATask,
    MultiRCTask,
    WiCTask,
    WSCTask,
)
from .util import (
    add_all_to_overall,
    add_mean_of_mean_to_overall,
    add_mean_to_overall,
    add_suffix,
)


def get_raw_task_datasets(data_dir):
    path = pathlib.Path(data_dir)
    glue_splits = ["train", "val", "test"]

    def to_dataset(splits, dataset_class):
        return {k: dataset_class(v) for k, v in splits.items()}

    def to_glue_dataset(name, dataset_class):
        return to_dataset(
            read_splits(path / name, glue_splits, ".jsonl"), dataset_class
        )

    return {
        "BoolQ": to_glue_dataset("BoolQ", FlatDataset),
        "CB": to_glue_dataset("CB", FlatDataset),
        "COPA": to_glue_dataset("COPA", FlatDataset),
        "MultiRC": to_glue_dataset("MultiRC", MultiRCDataset),
        "ReCoRD": to_glue_dataset("ReCoRD", ReCoRDDataset),
        "RTE": to_glue_dataset("RTE", FlatDataset),
        "WiC": to_glue_dataset("WiC", FlatDataset),
        "WSC": to_glue_dataset("WSC", FlatDataset),
    }


def get_pretrain_datasets(data_dir, tokenizer):
    path = pathlib.Path(data_dir)
    splits = ["train", "val"]
    cached_sizes_paths = {
        split: path / "wiki" / f"cached-sizes-{split}.npy" for split in splits
    }
    cached_sizes = {
        split: np.load(path) if path.exists() else None
        for split, path in cached_sizes_paths.items()
    }
    wiki = load_dataset("wikipedia", "20200501.en")
    wiki = wiki["train"].train_test_split(test_size=1000, seed=42)
    train_dataset = WikiDataset(
        wiki, "train", "text", tokenizer, cached_sizes=cached_sizes["train"]
    )
    val_dataset = MiniWikiDataset(wiki, "test", "text", tokenizer)
    return {"train": train_dataset, "val": val_dataset}


def get_tasks(tokenizer):
    task_ids = {
        "BoolQ": 2,
        "CB": 3,
        "COPA": 4,
        "MultiRC": 5,
        "ReCoRD": 6,
        "RTE": 7,
        "WiC": 8,
        "WSC": 9,
    }
    task_ids = {
        k: tokenizer.token_to_id("[CLS{}]".format(v)) for k, v in task_ids.items()
    }
    sep_id = tokenizer.token_to_id("[SEP]")
    mask_id = tokenizer.token_to_id("[MASK]")
    labels = {
        "BoolQ": [(False, sep_id), (True, task_ids["BoolQ"])],
        "CB": [
            ("contradiction", sep_id),
            ("entailment", task_ids["COPA"]),
            ("neutral", mask_id),
        ],
        "COPA": [(0, sep_id), (1, task_ids["COPA"])],
        "MultiRC": [(0, sep_id), (1, task_ids["MultiRC"])],
        "ReCoRD": sep_id,
        "RTE": [("not_entailment", sep_id), ("entailment", task_ids["RTE"])],
        "WiC": [(False, sep_id), (True, task_ids["WiC"])],
        "WSC": [(False, sep_id), (True, task_ids["WSC"])],
    }
    return {
        "BoolQ": BoolQTask(task_ids["BoolQ"], tokenizer, labels["BoolQ"]),
        "CB": CBTask(task_ids["CB"], tokenizer, labels["CB"]),
        "COPA": COPATask(task_ids["COPA"], tokenizer, labels["COPA"]),
        "MultiRC": MultiRCTask(task_ids["MultiRC"], tokenizer, labels["MultiRC"]),
        "ReCoRD": ReCoRDTask(task_ids["ReCoRD"], tokenizer, labels["ReCoRD"]),
        "RTE": RTETask(task_ids["RTE"], tokenizer, labels["RTE"]),
        "WiC": WiCTask(task_ids["WiC"], tokenizer, labels["WiC"]),
        "WSC": WSCTask(task_ids["WSC"], tokenizer, labels["WSC"]),
    }


class Scorer:
    superglue_names = ["BoolQ", "CB", "COPA", "MultiRC", "ReCoRD", "RTE", "WiC", "WSC"]
    dataset_names = superglue_names
    metrics_map = {
        "CB": ["CB_F1", "CB_Acc"],
        "MultiRC": ["MultiRC_F1a", "MultiRC_EM"],
        "ReCoRD": ["ReCoRD_F1", "ReCoRD_Acc"],
    }
    all_names = dataset_names + ["MLM"] + add_suffix(dataset_names + ["MLM"], "_loss")

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
        add_mean_of_mean_to_overall(overall, scores, cls.superglue_names, "SuperGLUE")
        add_mean_to_overall(
            overall, scores, add_suffix(cls.superglue_names, "_loss"), "SuperGLUE_loss"
        )
        add_all_to_overall(overall, scores, ["MLM", "MLM_loss"])
        add_mean_to_overall(overall, overall, ["MLM", "SuperGLUE"], "Overall")
        add_mean_to_overall(overall, overall, ["MLM_loss", "SuperGLUE_loss"], "loss")
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
        if any(name in score_names for name in cls.superglue_names):
            overall.append("SuperGLUE")

        if any(name + "_loss" in score_names for name in cls.superglue_names):
            overall.append("SuperGLUE_loss")

        for metric in ["MLM", "MLM_loss"]:
            if metric in score_names:
                overall.append(metric)

        return overall

    @staticmethod
    def get_maximize_metrics():
        return {
            "Overall": True,
            "SuperGLUE": True,
            "SuperGLUE_loss": False,
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
