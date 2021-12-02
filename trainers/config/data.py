from tokenizers import Tokenizer
from torch.utils.data import DataLoader

import data.config
from data.datasets import (
    MetaCollater,
    MetaSampler,
    PretrainTaskDataset,
    TaskCollater,
    TaskDataset,
    get_meta_dataset,
    get_task_datasets,
)
from data.tasks import get_mlm_task
from data.tokenizers import get_pretrain_tokenizer, get_task_tokenizer
from .util import bool_arg


def add_special_tokens(args, tokenizer):
    args.ignore_idx = -1
    args.padding_idx = tokenizer.padding["pad_id"]


def get_tokenizers(args):
    pretrain_tokenizer = get_pretrain_tokenizer(args.tokenizer_dir, args.max_positions)
    task_tokenizer = get_task_tokenizer(
        args.tokenizer_dir, args.max_positions, args.context_window_stride
    )
    add_special_tokens(args, pretrain_tokenizer)
    return pretrain_tokenizer, task_tokenizer


def get_task_loader(args, task_dataset, shuffle):
    return DataLoader(
        dataset=task_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=TaskCollater(args.padding_idx),
        shuffle=shuffle,
    )


def get_pretrain_datasets(
    args, task_tokenizer: Tokenizer, pretrain_tokenizer: Tokenizer
):
    if args.dataset == "nlp":
        data_config = data.config.nlp
        data_dir = args.data_dir
    elif args.dataset == "two_moons":
        data_config = data.config.two_moons
        data_dir = data_config.get_raw_data(args.two_moons_finetune_samples)
    else:
        raise ValueError("Unrecognized dataset: {}".format(args.dataset))

    raw_datasets = data_config.get_raw_task_datasets(data_dir)
    tasks = data_config.get_tasks(task_tokenizer)
    task_datasets = get_task_datasets(raw_datasets, tasks, mini_val_size=args.val_size)
    pretrain_datasets = data_config.get_pretrain_datasets(
        data_dir, pretrain_tokenizer, mini_val_size=args.val_size
    )
    mlm_task = get_mlm_task(
        pretrain_tokenizer, args.mask_prob, args.unmask_prob, args.randomize_prob
    )
    pretrain_task_datasets = {
        split: PretrainTaskDataset(
            pretrain_dataset, mlm_task, window="random", strict=False
        )
        for split, pretrain_dataset in pretrain_datasets.items()
    }
    mlm_task_datasets = {
        "MLM": {
            split: TaskDataset(
                pretrain_dataset, mlm_task, window="random", strict=False
            )
            for split, pretrain_dataset in pretrain_datasets.items()
        }
    }

    if args.include_mlm_task:
        task_datasets.update(mlm_task_datasets)
        meta_dataset = get_meta_dataset(pretrain_task_datasets, task_datasets, "train")
    else:
        meta_dataset = get_meta_dataset(pretrain_task_datasets, task_datasets, "train")
        task_datasets.update(mlm_task_datasets)

    meta_sampler = MetaSampler(meta_dataset, args.epoch_size, args.samples_per_task)
    meta_loader = DataLoader(
        dataset=meta_dataset,
        sampler=meta_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=MetaCollater(args.padding_idx),
    )
    task_loaders = {
        name: {
            split: get_task_loader(args, dataset, shuffle=(split != "train"))
            for split, dataset in splits.items()
        }
        for name, splits in task_datasets.items()
    }

    return (
        meta_dataset,
        meta_loader,
        task_datasets,
        task_loaders,
    )


def get_finetune_datasets(args, task_tokenizer: Tokenizer):
    if args.dataset == "nlp":
        data_config = data.config.nlp
        data_dir = args.data_dir
    elif args.dataset == "two_moons":
        data_config = data.config.two_moons
        data_dir = data_config.get_raw_data(args.two_moons_finetune_samples)
    else:
        raise ValueError("Unrecognized dataset: {}".format(args.dataset))

    raw_datasets = data_config.get_raw_task_datasets(data_dir)
    tasks = data_config.get_tasks(task_tokenizer)
    task_datasets = get_task_datasets(raw_datasets, tasks, mini_val_size=args.val_size)
    task_loaders = {
        name: {
            split: get_task_loader(args, dataset, shuffle=(split != "train"))
            for split, dataset in splits.items()
        }
        for name, splits in task_datasets.items()
    }
    return task_datasets, task_loaders


def get_scorer(args):
    return get_scorer_from_name(args.dataset)


def get_scorer_from_name(name):
    if name == "nlp":
        data_config = data.config.nlp
    elif name == "two_moons":
        data_config = data.config.two_moons
    else:
        raise ValueError("Unrecognized dataset: {}".format(name))
    return data_config.Scorer()


def get_all_dataset_names():
    return ["nlp", "two_moons"]


def get_dataset_maximize_metric():
    maximize_metric = {}
    for scorer in [get_scorer_from_name(d) for d in get_all_dataset_names()]:
        maximize_metric.update(scorer.get_maximize_metrics())
    return maximize_metric


def get_all_dataset_overall_names():
    names = []
    for scorer in [get_scorer_from_name(d) for d in get_all_dataset_names()]:
        names += scorer.get_overall_names()
    return list(set(names))


def add_tokenizer_args(parser):
    parser.add_argument(
        "--max_positions",
        type=int,
        default=512,
        help="Maximum number of tokens.",
    )
    parser.add_argument(
        "--context_window_stride",
        type=int,
        default=256,
        help="Stride for selecting sliding windows from the context.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="./save/tokenizers/wiki-bpe/",
        help="Base directory for tokenizers",
    )


def add_data_args(parser):
    parser.add_argument(
        "--dataset",
        choices=get_all_dataset_names(),
        default=get_all_dataset_names()[0],
        help="Which dataset to load.",
    )
    parser.add_argument(
        "--two_moons_finetune_samples",
        type=int,
        default=10,
        help="How many samples to use for finetuning for the two_moons dataset",
    )
    parser.add_argument(
        "--include_mlm_task",
        type=bool_arg,
        default=False,
        help="Whether to include mlm as a target task",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./save/data/", help="Base directory for data"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of sub-processes to use per data loader.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.",
    )


def add_mlm_args(parser):
    parser.add_argument(
        "--mask_prob", type=float, default=0.15, help="Mask probability."
    )
    parser.add_argument(
        "--unmask_prob",
        type=float,
        default=0.1,
        help="Probability to leave mask unchanged.",
    )
    parser.add_argument(
        "--randomize_prob",
        type=float,
        default=0.1,
        help="Probability to use a random token instead of mask.",
    )
