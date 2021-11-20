# Copyright (c) Jeffrey Shen

import json
import argparse

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
    SimpleState,
    ModelState,
    ModelSaver,
)
from trainers.meta_utils import (
    forward_step,
    update_step,
)
from trainers.optimizers import NoopOptimizer
import trainers.config as config
import models.transformer as T


def get_stats(tbx, pretrain_tokenizer, scorer, args):
    train_tbx = stats.TensorboardScalars(
        tbx,
        "train",
        [
            "student.lr",
            "student.loss0",
            "steps",
        ],
    )
    val_tbx = stats.TensorboardScalars(
        tbx,
        "train",
        ["loss"]
        + scorer.get_overall_names()
        + scorer.get_metric_names([args.task])
        + [args.task + "_loss"],
    )
    student_tbx = stats.TensorboardWeights(tbx, "student")
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
    return train_tbx, val_tbx, student_tbx, text_tbxs


def train(args):
    args.save_dir = config.get_save_dir(args.save_dir, "train", args.name)
    device = config.update_gpus(args)
    log = stats.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    state = TrainerState(log=log)

    config.reload_checkpoint(args, state, log)

    # Always use the current save_dir and name
    args = KeepSimpleState.init_from(args, keep=["save_dir", "name"])
    state.track_object("args", args)
    log.info(f"Args: {json.dumps(vars(args), indent=4, sort_keys=True)}")

    # Set random seed
    rand = RandomState(args.seed)
    saver = config.get_model_saver(args, log)

    # Get tokenizers, data loaders, and visualizers
    log.info("Building dataset...")
    pretrain_tokenizer, task_tokenizer = config.get_tokenizers(args)
    task_datasets, task_loaders = config.get_finetune_datasets(args, task_tokenizer)
    scorer = config.get_scorer(args)
    train_tbx, val_tbx, student_tbx, text_tbxs = get_stats(
        tbx, pretrain_tokenizer, scorer, args
    )
    task_dataset, task_loader = (
        task_datasets[args.task][args.split],
        task_loaders[args.task][args.split],
    )
    val_dataset, val_loader = (
        task_datasets[args.task]["mini_val"],
        task_loaders[args.task]["mini_val"],
    )

    # Get model
    student = ModelState()
    log.info("Building model...")
    student.model = config.get_roberta_model(args, pretrain_tokenizer.get_vocab_size())
    if args.load_path and not state.is_reloading():
        log.info(f"Loading model from {args.load_path}...")
        student.model, _ = ModelSaver.load_model(
            student.model, args.load_path, device, strict=False
        )
    student.model = student.model.to(device)
    state.track_object("student.model", student.model)
    log.info("Student model:")
    log.info(student.model)

    # Get optimizer, scheduler, and scaler
    total_steps = (
        max(1, args.num_epochs)
        * len(task_dataset)
        // args.batch_size
        // args.gradient_accumulation
    )

    # TODO: Use the same settings for now
    student.optimizer = config.get_adamw_optimizer(args, student.model)
    student.noop_optimizer = NoopOptimizer(student.model.parameters())
    student.scheduler = config.get_lwpd_scheduler(args, student.optimizer, total_steps)
    student.scaler = amp.GradScaler()

    state.track_object("student.optimizer", student.optimizer)
    state.track_object("student.scheduler", student.scheduler)
    state.track_object("student.scaler", student.scaler)

    # Train
    step = SimpleState(
        epoch=0, step_num=0, sample_num=0, samples_til_eval=args.eval_per_n_samples
    )
    state.track_object("step", step)
    state.track_object("random", rand)
    assert not state.is_reloading()

    log.info("Training...")
    student.model.train()

    while step.epoch != args.num_epochs:
        config.save_checkpoint(args, state)
        step.epoch += 1
        log.info(f"Starting epoch {step.epoch}...")
        student_tbx(student.model, step.epoch)

        with torch.enable_grad(), tqdm(total=len(task_loader)) as progress_bar:
            for (idxs, x, y) in task_loader:
                x = x.to(device)
                y = y.to(device)

                info = train_step(x, y, student, args, step)

                # Log info
                progress_bar.update(1)
                progress_bar.set_postfix(
                    epoch=step.epoch,
                    loss=info["student.loss0"],
                )
                train_tbx(info, step.sample_num)

                if step.samples_til_eval <= 0:
                    # Evaluate and save checkpoint
                    log.info(f"Evaluating at sample step {step.sample_num}...")
                    losses, val_scores, tensors, preds = evaluate(
                        student.model, val_loader, val_dataset, device, args
                    )
                    for name in text_tbxs:
                        for idx in tensors[name]:
                            tensor_x, tensor_y = tensors[name][idx]
                            text_tbxs[name].add([idx, tensor_x, tensor_y, preds[name][idx]])
                        text_tbxs[name](step.sample_num)
                        text_tbxs[name].clear()

                    overall = scorer.scores_to_overall(val_scores)
                    for k in overall:
                        overall[k] *= 100
                    overall["loss"] = np.mean(list(losses.values()))

                    saver.save(step.sample_num, student.model, overall)

                    # Log to console
                    overall_str = ", ".join(
                        f"{k}: {v:05.2f}" for k, v in overall.items()
                    )
                    log.info(f"Val {overall_str}")

                    # Log to TensorBoard
                    log.info("Visualizing in TensorBoard...")
                    metrics = scorer.scores_to_metrics(val_scores)
                    for k in metrics:
                        metrics[k] *= 100
                    metrics.update(overall)
                    metrics.update({k + "_loss": v for k, v in losses.items()})

                    val_tbx(metrics, step.sample_num)


def train_step(x, y, student, args, step):
    info = {}
    batch_size = x.size(0)
    info["batch_size"] = batch_size

    # Student x, y forward, backward pass
    forward_step(student, x, y, args, info)

    # Step student
    update_step(student, step, batch_size, args, info)

    info["student.lr"] = student.optimizer.param_groups[0]["lr"]

    return info


def evaluate(model, val_loader, val_dataset, device, args):
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
        val_scores[name] = val_dataset.score(preds[name])

    model.train()
    losses = {name: loss_meter.avg for name, loss_meter in loss_meters.items()}

    return losses, val_scores, tensors, preds


def add_train_args(parser: argparse.ArgumentParser):
    add_train_test_args(parser)
    config.add_train_args(parser)
    config.add_model_saver_args(
        parser, metric_names=["loss"] + config.get_all_dataset_overall_names()
    )
    config.add_lwpd_scheduler_args(parser)
    config.add_adamw_optimizer_args(parser)

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
        "--eval_per_n_samples",
        type=int,
        default=10000,
        help="Number of samples between successive evaluations.",
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
        help="Which task to finetune on.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to use for training.",
    )


def add_train_test_args(parser):
    config.add_tokenizer_args(parser)
    config.add_data_args(parser)
    config.add_roberta_args(parser)
    config.add_train_test_args(parser)
