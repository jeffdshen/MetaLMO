# Copyright (c) Jeffrey Shen

import json
import argparse
import pathlib

import torch
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

import trainers.stats as stats
from trainers.state import (
    KeepSimpleState,
    TrainerState,
    RandomState,
    ModelState,
    ModelSaver,
)
import trainers.config as config
import models.transformer as T


def get_stats(tbx, pretrain_tokenizer, args):
    formatter = stats.JoinTextFormatter(
        [
            stats.StrTextFormatter(["idx"]),
            stats.TokenizedTextFormatter(pretrain_tokenizer, ["x", "y"]),
            stats.StrTextFormatter(["pred"]),
        ]
    )
    text_tbxs = {
        name: stats.TensorboardText(
            tbx, "val", "example_{}".format(name), formatter, args.num_visuals
        )
        for name in [args.task]
    }
    return text_tbxs


def train(args):
    args.save_dir = config.get_save_dir(args.save_dir, args.name)
    device = config.update_gpus(args)
    log = stats.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)

    # Always use the current save_dir and name
    args = KeepSimpleState.init_from(args, keep=["save_dir", "name"])
    log.info(f"Args: {json.dumps(vars(args), indent=4, sort_keys=True)}")

    # Set random seed
    rand = RandomState(args.seed)

    # Get tokenizers, data loaders, and visualizers
    log.info("Building dataset...")
    pretrain_tokenizer, task_tokenizer = config.get_tokenizers(args)
    task_datasets, task_loaders = config.get_finetune_datasets(args, task_tokenizer)
    scorer = config.get_scorer(args)
    text_tbxs = get_stats(tbx, pretrain_tokenizer, args)
    task_dataset, task_loader = (
        task_datasets[args.task][args.split],
        task_loaders[args.task][args.split],
    )

    # Get model
    student = ModelState()
    log.info("Building model...")
    student.model = config.get_roberta_model(args, pretrain_tokenizer.get_vocab_size())
    if args.load_path:
        log.info(f"Loading model from {args.load_path}...")
        student.model, step_num = ModelSaver.load_model(
            student.model, args.load_path, device, strict=False
        )
        log.info(f"Loaded model from step {step_num}")
    student.model = student.model.to(device)
    log.info("Student model:")
    log.info(student.model)

    # Evaluate
    log.info(f"Evaluating on {args.split} split...")
    student.model.eval()

    losses, val_scores, tensors, preds = evaluate(
        student.model, task_loader, task_dataset, device, args
    )
    for name in text_tbxs:
        for idx in tensors[name]:
            tensor_x, tensor_y = tensors[name][idx]
            text_tbxs[name].add([idx, tensor_x, tensor_y, preds[name][idx]])
        text_tbxs[name](0)
        text_tbxs[name].clear()

    overall = scorer.scores_to_overall(val_scores)
    for k in overall:
        overall[k] *= 100
    overall["loss"] = np.mean(list(losses.values()))
    metrics = scorer.scores_to_metrics(val_scores)
    for k in metrics:
        metrics[k] *= 100
    metrics.update(overall)
    metrics.update({k + "_loss": v for k, v in losses.items()})

    # Log to console
    metrics_str = ", ".join(f"{k}: {v:05.2f}" for k, v in metrics.items())
    log.info(f"{metrics_str}")

    for name in preds:
        pred_path = pathlib.Path(args.save_dir, name + ".jsonl")
        with open(pred_path, "w") as file:
            for idx, label in preds[name].items():
                file.write(json.dumps({"idx": idx, "label": label}))
                file.write("\n")


def evaluate(model, val_loader, val_dataset, device, args):
    # TODO: progress bar, merge with other evaluate functions
    model.eval()
    loss_meters = {}
    tensors = {}
    preds = {}
    val_scores = {}
    name = args.task
    with torch.no_grad():
        preds[name] = {}
        tensors[name] = {}
        loss_meters[name] = stats.AverageMeter()
        for idxs, x, y in val_loader:
            batch_size = x.size(0)
            x, y = x.to(device), y.to(device)
            mask = T.get_padding_mask(x, args.padding_idx)
            with amp.autocast(enabled=args.autocast):
                scores = model(x, padding_mask=mask)
                loss = model.get_loss(scores, y, mask)
            loss_meters[name].add(loss.item(), batch_size)
            pred = val_dataset.predict(idxs, x, scores)
            preds[name].update(pred)
            tensors[name].update(stats.tensors_groupby_flatten(idxs, [x, y]))
        if args.split != "test":
            val_scores[name] = val_dataset.score(preds[name])
        else:
            val_scores[name] = 0.0

    model.train()
    losses = {name: loss_meter.avg for name, loss_meter in loss_meters.items()}

    return losses, val_scores, tensors, preds


def add_train_args(parser: argparse.ArgumentParser):
    # TODO rename everything to run() and add run_args()
    add_train_test_args(parser)
    parser.add_argument(
        "--autocast",
        type=config.bool_arg,
        default=True,
        help="Turn on autocast everywhere.",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=128,
        help="Number of samples per task for validation.",
    )
    parser.add_argument(
        "--num_visuals",
        type=int,
        default=10,
        help="Number of examples to visualize in TensorBoard.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Which task to predict on.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to predict on.",
    )


def add_train_test_args(parser):
    config.add_tokenizer_args(parser)
    config.add_data_args(parser)
    config.add_roberta_args(parser)
    config.add_train_test_args(parser)
