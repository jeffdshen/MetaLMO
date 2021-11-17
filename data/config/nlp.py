import pathlib

import numpy as np
from data.datasets import (
    FlatDataset,
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


def get_pretrain_dataset(data_dir, tokenizer):
    path = pathlib.Path(data_dir)
    cached_sizes_path = path / "wiki" / "cached-sizes.npy"
    if cached_sizes_path.exists():
        cached_sizes = np.load(cached_sizes_path)
    else:
        cached_sizes = None
    wiki = load_dataset("wikipedia", "20200501.en")
    wiki_dataset = WikiDataset(
        wiki, "train", "text", tokenizer, cached_sizes=cached_sizes
    )
    return wiki_dataset


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
    @staticmethod
    def scores_to_overall(scores):
        superglue = ["BoolQ", "CB", "COPA", "MultiRC", "ReCoRD", "RTE", "WiC", "WSC"]
        superglue_score = np.mean([np.mean(scores[name]) for name in superglue])
        return {"Overall": superglue_score, "SuperGLUE": superglue_score}

    @staticmethod
    def scores_to_metrics(scores):
        metrics = scores.copy()
        if "CB" in metrics:
            f1, acc = metrics.pop("CB")
            metrics["CB_F1"] = f1
            metrics["CB_Acc"] = acc

        if "MultiRC" in metrics:
            f1a, em = metrics.pop("MultiRC")
            metrics["MultiRC_F1a"] = f1a
            metrics["MultiRC_EM"] = em

        if "ReCoRD" in metrics:
            f1, acc = metrics.pop("ReCoRD")
            metrics["ReCoRD_F1"] = f1
            metrics["ReCoRD_Acc"] = acc

        return metrics

    @staticmethod
    def get_overall_names():
        return ["Overall", "SuperGLUE"]

    @staticmethod
    def get_maximize_metrics():
        return {
            "Overall": True,
            "SuperGLUE": True,
        }

    @staticmethod
    def get_metric_names():
        return [
            "BoolQ",
            "CB_F1",
            "CB_Acc",
            "COPA",
            "MultiRC_F1a",
            "MultiRC_EM",
            "ReCoRD_F1",
            "ReCoRD_Acc",
            "RTE",
            "WiC",
            "WSC",
        ]
