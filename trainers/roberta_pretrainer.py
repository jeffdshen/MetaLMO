# Copyright (c) Jeffrey Shen

import json
import argparse

import torch
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import trainers.stats as stats
from trainers.state import (
    KeepSimpleState,
    TrainerState,
    RandomState,
    SimpleState,
    ModelState,
)
from trainers.meta_utils import (
    pseudo_step,
    query_step,
    real_step,
    support_step,
    update_step,
)
from trainers.optimizers import NoopOptimizer
import trainers.config as config
import models.transformer as T


def get_stats(tbx, pretrain_tokenizer, args):
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
        [
            "student.lr",
            "unscaled_h",
            "h",
            "student.loss0",
            "student.loss1",
            "student.loss2",
            "steps",
        ],
    )
    student_tbx = stats.TensorboardWeights(tbx, "student")
    formatter = stats.TokenizedTextFormatter(
        pretrain_tokenizer, ["x_m", "y_m", "y_hat"]
    )
    mlm_tbx = stats.TensorboardText(tbx, "train", "mlm", formatter, args.num_visuals)
    return train_tbx, val_tbx, student_tbx, mlm_tbx


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
    task_datasets, val_task_loaders, meta_dataset, meta_loader = config.get_datasets(
        args, task_tokenizer, pretrain_tokenizer
    )
    train_tbx, val_tbx, student_tbx, mlm_tbx = get_stats(tbx, pretrain_tokenizer, args)

    # Get model
    student = ModelState()
    log.info("Building model...")
    student.model = config.get_roberta_model(args, pretrain_tokenizer.get_vocab_size())
    # if args.load_path and not state.is_reloading():
    #     log.info(f"Loading model from {args.load_path}...")
    #     model, _ = ModelSaver.load_model(model, args.load_path, device, strict=False)
    student.model = student.model.to(device)
    state.track_object("student.model", student.model)
    log.info("Student model:")
    log.info(student.model)

    # Get optimizer, scheduler, and scaler
    total_steps = (
        max(1, args.num_epochs)
        * args.epoch_size
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
        epoch=0,
        step_num=0,
        sample_num=0,
        samples_til_log=args.log_per_n_samples,
        samples_til_eval=args.eval_per_n_samples,
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

        with torch.enable_grad(), tqdm(total=len(meta_loader)) as progress_bar:
            for (_, x_u, _, x_m, y_m), (_, x_s, y_s), (_, x_q, y_q) in meta_loader:
                x_u = x_u.to(device)
                x_m = x_m.to(device)
                y_m = y_m.to(device)
                x_s = x_s.to(device)
                y_s = y_s.to(device)
                x_q = x_q.to(device)
                y_q = y_q.to(device)

                if step.samples_til_log > 0:
                    info = train_step(
                        x_u, x_m, y_m, x_s, y_s, x_q, y_q, student, args, step
                    )
                    mlm_tbx.add_all(info["mlm"])
                    train_tbx(info, step.sample_num)
                else:
                    log.info(f"Evaluating at sample step {step.sample_num}...")
                    info = log_step(
                        x_u, x_m, y_m, x_s, y_s, x_q, y_q, student, args, step
                    )
                    overall = {
                        "loss0": info["student.loss0"],
                        "loss2": info["student.loss2"],
                    }
                    saver.save(step.sample_num, student.model, overall)

                    # Log to console
                    overall_str = ", ".join(
                        f"{k}: {v:05.2f}" for k, v in overall.items()
                    )
                    log.info(f"Val {overall_str}")

                    log.info("Visualizing in TensorBoard...")
                    val_tbx(info, step.sample_num)

                if step.samples_til_eval <= 0:
                    mlm_tbx(step.sample_num)
                    mlm_tbx.clear()

                # Log info
                progress_bar.update(1)
                progress_bar.set_postfix(
                    epoch=step.epoch,
                    loss=info["student.loss0"],
                )


def train_step(x_u, x_m, y_m, x_s, y_s, x_q, y_q, student, args, step):
    info = {}
    batch_size = x_u.size(0)
    info["batch_size"] = batch_size

    # Student x_m, y_m forward, backward pass
    real_step(student, x_m, y_m, args, info)

    # Step student
    update_step(student, step, batch_size, args, info)

    if step.samples_til_log <= 0:
        step.samples_til_log = args.log_per_n_samples
    step.samples_til_log -= batch_size

    info["student.lr"] = student.optimizer.param_groups[0]["lr"]

    return info


def log_step(x_u, x_m, y_m, x_s, y_s, x_q, y_q, student, args, step):
    info = {}
    batch_size = x_u.size(0)
    info["batch_size"] = batch_size

    # Student x_hat, y_hat forward, backward pass
    deltas = pseudo_step(student, x_m, y_m, args, info)

    # Student x_l, y_l forward, backward pass, compute h
    orig_param = support_step(student, x_s, y_s, args, info)
    h = query_step(student, x_q, y_q, orig_param, deltas, args, info)

    # Step student
    update_step(student, step, batch_size, args, info)
    if step.samples_til_log <= 0:
        step.samples_til_log = args.log_per_n_samples
    step.samples_til_log -= batch_size

    info["student.lr"] = student.optimizer.param_groups[0]["lr"]

    return info


def add_train_args(parser: argparse.ArgumentParser):
    add_train_test_args(parser)
    config.add_train_args(parser)
    config.add_model_saver_args(parser, metric_names=["loss0", "loss2"])
    config.add_lwpd_scheduler_args(parser)
    config.add_adamw_optimizer_args(parser)

    parser.add_argument(
        "--autocast",
        type=config.bool_arg,
        default=True,
        help="Turn on autocast everywhere.",
    )
    parser.add_argument(
        "--inner_lr", type=float, default=0.1, help="Inner learning rate for student."
    )
    parser.add_argument(
        "--samples_per_task",
        type=int,
        default=16,
        help="How many times to repeat the task before sampling from a different one.",
    )
    parser.add_argument(
        "--epoch_size", type=int, default=20000, help="Number of samples per epoch."
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=128,
        help="Number of samples per task for validation.",
    )
    parser.add_argument(
        "--log_per_n_samples",
        type=int,
        default=1000,
        help="Number of samples between h evaluations",
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


def add_train_test_args(parser):
    config.add_tokenizer_args(parser)
    config.add_data_args(parser)
    config.add_mlm_args(parser)
    config.add_roberta_args(parser)
    config.add_train_test_args(parser)
