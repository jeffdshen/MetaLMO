# Copyright (c) Jeffrey Shen

import os
import json
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import trainers.stats as stats
from trainers.state import (
    KeepSimpleState,
    TrainerState,
    RandomState,
    SimpleState,
    ModelSaver,
)
from trainers.optimizers import NoopOptimizer
import trainers.config as config
from data.tokenizers import get_pretrain_tokenizer, get_task_tokenizer
from data.datasets import (
    get_meta_dataset,
    get_pretrain_dataset,
    get_raw_task_datasets,
    get_task_datasets,
    MetaSampler,
    MetaCollater,
    TaskCollater,
)
from data.tasks import get_tasks, scores_to_overall, scores_to_metrics
import models.transformer as T


def get_task_loader(args, task_dataset):
    return DataLoader(
        dataset=task_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=TaskCollater(args.padding_idx),
    )


def get_datasets(args, task_tokenizer, pretrain_tokenizer):
    raw_datasets = get_raw_task_datasets(args.data_dir)
    tasks = get_tasks(task_tokenizer)
    task_datasets = get_task_datasets(raw_datasets, tasks, mini_val_size=args.val_size)
    pretrain_dataset = get_pretrain_dataset(args.data_dir, pretrain_tokenizer)
    meta_dataset = get_meta_dataset(pretrain_dataset, task_datasets, "train")
    meta_sampler = MetaSampler(meta_dataset, args.epoch_size)
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


@dataclass
class ModelState:
    model = None
    optimizer = None
    scheduler = None
    scaler = None
    noop_optimizer = None


def get_stats(tbx):
    train_tbx = stats.TensorboardScalars(
        tbx,
        "train",
        [
            "student.lr",
            "teacher.lr",
            "h",
            "teacher.loss_l",
            "teacher.loss_u",
            "student.loss0",
            "student.loss1",
            "steps",
        ],
    )
    val_tbx = stats.TensorboardScalars(
        tbx,
        "val",
        [
            "loss",
            "Overall",
            "SuperGLUE",
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
        ],
    )
    student_tbx = stats.TensorboardWeights(tbx, "student")
    return train_tbx, val_tbx, student_tbx


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

    # Get tokenizers
    pretrain_tokenizer = get_pretrain_tokenizer(args.tokenizer_dir, args.max_positions)
    task_tokenizer = get_task_tokenizer(
        args.tokenizer_dir, args.max_positions, args.context_window_stride
    )
    config.add_special_tokens(args, pretrain_tokenizer)

    # Visualizers
    # TODO: Visualize teacher questions
    train_tbx, val_tbx, student_tbx = get_stats(tbx)

    # Get data loader
    log.info("Building dataset...")
    task_datasets, val_task_loaders, meta_dataset, meta_loader = get_datasets(
        args, task_tokenizer, pretrain_tokenizer
    )

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
    teacher = ModelState()
    teacher.model = config.get_teacher_model(args, pretrain_tokenizer.get_vocab_size())
    teacher.model = teacher.model.to(device)
    state.track_object("teacher.model", teacher.model)
    log.info("Teacher model:")
    log.info(teacher.model)

    # Get optimizer, scheduler, and scaler
    total_steps = (
        max(1, args.num_epochs)
        * args.epoch_size
        // args.batch_size
        // args.gradient_accumulation
    )

    # TODO: Use the same settings for now
    student.optimizer = config.get_adamw_optimizer(args, student.model)
    student.noop_optimizer = NoopOptimizer(student.model.parameters())
    student.scheduler = config.get_lwpd_scheduler(args, student.optimizer, total_steps)
    student.scaler = amp.GradScaler()
    teacher.optimizer = config.get_adamw_optimizer(args, teacher.model)
    teacher.scheduler = config.get_lwpd_scheduler(args, teacher.optimizer, total_steps)
    teacher.scaler = amp.GradScaler()

    state.track_object("student.optimizer", student.optimizer)
    state.track_object("student.scheduler", student.scheduler)
    state.track_object("student.scaler", student.scaler)
    state.track_object("teacher.optimizer", teacher.optimizer)
    state.track_object("teacher.scheduler", teacher.scheduler)
    state.track_object("teacher.scaler", teacher.scaler)

    # Train
    step = SimpleState(
        epoch=0, step_num=0, sample_num=0, samples_til_eval=args.eval_per_n_samples
    )
    state.track_object("step", step)
    state.track_object("random", rand)
    assert not state.is_reloading()

    log.info("Training...")
    student.model.train()
    teacher.model.train()

    while step.epoch != args.num_epochs:
        config.save_checkpoint(args, state)
        step.epoch += 1
        log.info(f"Starting epoch {step.epoch}...")
        student_tbx(student.model, step.epoch)

        with torch.enable_grad(), tqdm(total=len(meta_loader)) as progress_bar:
            for x_u, (idxs, x_l, y_l) in meta_loader:
                x_u = x_u.to(device)
                x_l = x_l.to(device)
                y_l = y_l.to(device)

                info = train_step(x_u, x_l, y_l, student, teacher, args, step)

                # Log info
                progress_bar.update(1)
                progress_bar.set_postfix(epoch=step.epoch, loss=info["student.loss1"])
                train_tbx(info, step.sample_num)

                if step.samples_til_eval <= 0:
                    # Evaluate and save checkpoint
                    log.info(f"Evaluating at sample step {step.sample_num}...")
                    loss, val_scores, preds = evaluate(
                        student.model, val_task_loaders, task_datasets, device, args
                    )
                    overall = scores_to_overall(val_scores)
                    for k in overall:
                        overall[k] *= 100
                    overall["loss"] = loss
                    saver.save(step.sample_num, student.model, overall)

                    # Log to console
                    overall_str = ", ".join(
                        f"{k}: {v:05.2f}" for k, v in overall.items()
                    )
                    log.info(f"Val {overall_str}")

                    # Log to TensorBoard
                    log.info("Visualizing in TensorBoard...")
                    metrics = scores_to_metrics(val_scores)
                    for k in metrics:
                        metrics[k] *= 100
                    metrics.update(overall)

                    val_tbx(metrics, step.sample_num)
                    # TODO: visualize teacher questions


def train_step(x_u, x_l, y_l, student, teacher, args, step, autocast=True):
    info = {}
    batch_size = x_u.size(0)
    info["batch_size"] = batch_size

    # sample x_hat, y_hat, compute teacher_grad
    mask_x_u = T.get_padding_mask(x_u, args.padding_idx)
    with amp.autocast(enabled=autocast):
        scores_x, scores_y = teacher.model(x_u, padding_mask=mask_x_u)
        x_hat, y_hat = teacher.model.sample(
            scores_x, scores_y, x_u, x_u, mask_x_u, mask_x_u, k=1
        )
        x_hat, y_hat = x_hat.squeeze(0), y_hat.squeeze(0)
        loss = teacher.model.get_loss(
            scores_x, scores_y, x_hat, y_hat, mask_x_u, mask_x_u
        )
    info["teacher.loss_u"] = loss.item()
    teacher_grad = autograd.grad(
        teacher.scaler.scale(loss / args.gradient_accumulation),
        teacher.model.parameters(),
    )

    # Student x_hat, y_hat forward, backward pass
    deltas = [param.data.clone() for param in student.model.parameters()]
    mask_x_hat = T.get_padding_mask(x_hat, args.padding_idx)
    with amp.autocast(enabled=autocast):
        scores = student.model(x_hat, padding_mask=mask_x_hat)
        loss = student.model.get_loss(scores, y_hat, mask_x_hat)
    info["student.loss0"] = loss.item()
    student.scaler.scale(loss).backward()
    student.scaler.unscale_(student.optimizer)
    nn.utils.clip_grad_norm_(student.model.parameters(), args.max_grad_norm)
    student.scaler.step(student.optimizer)
    student.scaler.update()
    student.scheduler.step()
    student.optimizer.zero_grad()
    for delta, param in zip(deltas, student.model.parameters()):
        delta -= param.data

    # Student x_l, y_l forward, backward pass, compute h
    mask_x_l = T.get_padding_mask(x_l, args.padding_idx)
    with amp.autocast(enabled=autocast):
        scores = student.model(x_l, padding_mask=mask_x_l)
        loss = student.model.get_loss(scores, y_l, mask_x_l)
    info["student.loss1"] = loss.item()
    student.scaler.scale(loss).backward()
    student.scaler.unscale_(student.noop_optimizer)
    nn.utils.clip_grad_norm_(student.model.parameters(), args.max_grad_norm)
    student.scaler.step(student.noop_optimizer)
    student.scaler.update()
    h = 0.0
    for delta, param in zip(deltas, student.model.parameters()):
        h += delta.flatten().dot(param.grad.flatten())
    student.noop_optimizer.zero_grad()
    info["h"] = h.item()

    # Update teacher grad
    for grad, param in zip(teacher_grad, teacher.model.parameters()):
        if param.grad is None:
            param.grad = h * grad
        else:
            param.grad += h * grad

    # Teacher on labeled data
    mask_x_l = T.get_padding_mask(x_l, args.padding_idx)
    with amp.autocast(enabled=autocast):
        scores_x, scores_y = teacher.model(x_l, padding_mask=mask_x_l)
        loss = teacher.model.get_loss(scores_x, scores_y, x_l, y_l, mask_x_l, mask_x_l)
    info["teacher.loss_l"] = loss.item()
    teacher.scaler.scale(loss / args.gradient_accumulation).backward()

    # Teacher optimizer
    if (step.step_num + 1) % args.gradient_accumulation == 0:
        teacher.scaler.unscale_(teacher.optimizer)
        nn.utils.clip_grad_norm_(teacher.model.parameters(), args.max_grad_norm)
        teacher.scaler.step(teacher.optimizer)
        teacher.scaler.update()
        teacher.scheduler.step()
        teacher.optimizer.zero_grad()

    # Update steps
    step.step_num += 1
    step.sample_num += batch_size
    if step.samples_til_eval <= 0:
        step.samples_til_eval = args.eval_per_n_samples
    step.samples_til_eval -= batch_size
    info["steps"] = step.step_num / args.gradient_accumulation
    info["student.lr"] = student.optimizer.param_groups[0]["lr"]
    info["teacher.lr"] = student.optimizer.param_groups[0]["lr"]
    return info


def evaluate(model, val_loaders, task_datasets, device, args, autocast=True):
    loss_meter = stats.AverageMeter()

    model.eval()
    preds = {}
    val_scores = {}
    with torch.no_grad():
        for name, val_loader in val_loaders.items():
            task_dataset = task_datasets[name]["mini_val"]
            preds[name] = {}
            for idxs, x, y in val_loader:
                batch_size = x.size(0)
                x, y = x.to(device), y.to(device)
                mask = T.get_padding_mask(x, args.padding_idx)
                with amp.autocast(enabled=autocast):
                    scores = model(x, padding_mask=mask)
                    loss = model.get_loss(scores, y, mask)
                loss_meter.add(loss.item(), batch_size)
                pred = task_dataset.predict(idxs, x, scores)
                preds[name].update(pred)
            val_scores[name] = task_dataset.score(preds[name])

    model.train()
    return loss_meter.avg, val_scores, preds


def add_train_args(parser):
    add_train_test_args(parser)
    config.add_train_args(parser)
    config.add_model_saver_args(parser, metric_names=["loss", "Overall", "SuperGLUE"])
    config.add_lwpd_scheduler_args(parser)
    config.add_adamw_optimizer_args(parser)

    parser.add_argument(
        "--epoch_size", type=int, default=25000, help="Number of samples per epoch."
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
        default=12500,
        help="Number of samples between successive evaluations.",
    )


def add_train_test_args(parser):
    config.add_tokenizer_args(parser)
    config.add_data_args(parser)
    config.add_roberta_args(parser)
    config.add_train_test_args(parser)