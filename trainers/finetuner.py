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
from trainers.util import (
    cat_pred_examples,
    optim_step,
    real_step,
    score_evaluate,
    update_step,
    evaluate,
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
    val_tasks = [args.task]
    val_score_names = [name + "_loss" for name in val_tasks]
    val_tbx = stats.TensorboardScalars(
        tbx,
        "val",
        scorer.get_overall_names(val_score_names)
        + scorer.get_metric_names(val_score_names),
    )
    student_tbx = stats.TensorboardWeights(tbx, "student")
    formatter = stats.TokenizedTextFormatter(
        pretrain_tokenizer, ["idx", "x", "y", "pred"]
    )
    text_tbxs = stats.TensorboardTexts(
        tbx, "val", "example_{}", [args.task], formatter, args.num_visuals
    )
    return train_tbx, val_tbx, student_tbx, text_tbxs


def train(args):
    args.save_dir = config.get_save_dir(args.save_dir, args.name)
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

    # Get model
    student = ModelState()
    log.info("Building model...")
    student.model = config.get_roberta_model(args, pretrain_tokenizer.get_vocab_size())
    if args.load_path and not state.is_reloading():
        log.info(f"Loading model from {args.load_path}...")
        student.model, step_num = ModelSaver.load_model(
            student.model, args.load_path, device, strict=False
        )
        log.info(f"Loaded model from step {step_num}")
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
                        student.model,
                        task_loaders,
                        task_datasets,
                        [args.task],
                        "mini_val",
                        device,
                        args,
                    )
                    overall, overall_str, metrics = score_evaluate(
                        scorer, val_scores, losses
                    )
                    # Log to console
                    log.info(f"Val {overall_str}")

                    # Log to TensorBoard
                    log.info("Visualizing in TensorBoard...")
                    text_tbxs(cat_pred_examples(tensors, preds), step)
                    val_tbx(metrics, step.sample_num)

                    # Save
                    saver.save(step.sample_num, student.model, overall)


def train_step(x, y, student, args, step):
    info = {}
    batch_size = x.size(0)
    info["batch_size"] = batch_size

    # Student x, y forward, backward pass
    real_step(student, x, y, args, info)

    # Step student
    optim_step(student, step, args)
    update_step(step, batch_size, args, info)

    info["student.lr"] = student.optimizer.param_groups[0]["lr"]

    return info


def add_train_args(parser: argparse.ArgumentParser):
    add_train_test_args(parser)
    config.add_train_args(parser)
    config.add_model_saver_args(
        parser, metric_names=config.get_all_dataset_overall_names()
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
