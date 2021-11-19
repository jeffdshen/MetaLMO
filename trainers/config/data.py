from tokenizers import Tokenizer
from torch.utils.data import DataLoader

import data.config
from data.datasets import (
    MetaCollater,
    MetaSampler,
    PretrainTaskDataset,
    TaskCollater,
    get_meta_dataset,
    get_task_datasets,
)
from data.tasks import get_mlm_task
from data.tokenizers import get_pretrain_tokenizer, get_task_tokenizer


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


def get_task_loader(args, task_dataset):
    return DataLoader(
        dataset=task_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=TaskCollater(args.padding_idx),
    )


def get_datasets(args, task_tokenizer: Tokenizer, pretrain_tokenizer: Tokenizer):
    if args.dataset == "nlp":
        data_config = data.config.nlp
        data_dir = args.data_dir
    elif args.dataset == "two_moons":
        data_config = data.config.two_moons
        data_dir = data_config.get_raw_data()
    else:
        raise ValueError("Unrecognized dataset: {}".format(args.dataset))

    raw_datasets = data_config.get_raw_task_datasets(data_dir)
    tasks = data_config.get_tasks(task_tokenizer)
    task_datasets = get_task_datasets(raw_datasets, tasks, mini_val_size=args.val_size)
    pretrain_dataset = data_config.get_pretrain_dataset(data_dir, pretrain_tokenizer)
    mlm_task = get_mlm_task(
        pretrain_tokenizer, args.mask_prob, args.unmask_prob, args.randomize_prob
    )
    pretrain_task_dataset = PretrainTaskDataset(
        pretrain_dataset, mlm_task, window="random", strict=False
    )
    meta_dataset = get_meta_dataset(pretrain_task_dataset, task_datasets, "train")
    meta_sampler = MetaSampler(meta_dataset, args.epoch_size, args.samples_per_task)
    meta_loader = DataLoader(
        dataset=meta_dataset,
        sampler=meta_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=MetaCollater(args.padding_idx),
    )
    val_task_loaders = {
        name: get_task_loader(args, splits["mini_val"])
        for name, splits in task_datasets.items()
    }
    return task_datasets, val_task_loaders, meta_dataset, meta_loader


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