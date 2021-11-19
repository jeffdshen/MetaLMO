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
from trainers.optimizers import NoopOptimizer
import trainers.config as config
from trainers.meta_utils import (
    h_step,
    mlm_step,
    pseudo_step,
    query_step,
    sample_step,
    support_step,
    update_step,
)
import models.transformer as T


def get_stats(tbx, pretrain_tokenizer, scorer, args):
    train_tbx = stats.TensorboardScalars(
        tbx,
        "train",
        [
            "student.lr",
            "teacher.lr",
            "unscaled_h",
            "h",
            "teacher.loss_x_m",
            "teacher.loss_y_m",
            "teacher.loss_u",
            "student.loss0",
            "student.loss1",
            "student.loss2",
            "steps",
        ],
    )
    val_tbx = stats.TensorboardScalars(
        tbx,
        "val",
        ["loss"] + scorer.get_overall_names() + scorer.get_metric_names(),
    )
    student_tbx = stats.TensorboardWeights(tbx, "student")
    formatter = stats.TokenizedTextFormatter(
        pretrain_tokenizer, ["x_u", "xhat_u", "yhat_u"]
    )
    pseudo_tbx = stats.TensorboardText(
        tbx, "train", "pseudo", formatter, args.num_visuals
    )
    return train_tbx, val_tbx, student_tbx, pseudo_tbx


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
    scorer = config.get_scorer(args)
    train_tbx, val_tbx, student_tbx, pseudo_tbx = get_stats(
        tbx, pretrain_tokenizer, scorer, args
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
            for (_, x_u, _, x_m, y_m), (_, x_s, y_s), (_, x_q, y_q) in meta_loader:
                x_u = x_u.to(device)
                x_m = x_m.to(device)
                y_m = y_m.to(device)
                x_s = x_s.to(device)
                y_s = y_s.to(device)
                x_q = x_q.to(device)
                y_q = y_q.to(device)

                info = train_step(
                    x_u, x_m, y_m, x_s, y_s, x_q, y_q, student, teacher, args, step
                )
                pseudo_tbx.add_all(info["pseudo"])

                # Log info
                progress_bar.update(1)
                progress_bar.set_postfix(
                    epoch=step.epoch,
                    h=info["h"],
                    teacher_loss=info["teacher.loss_y_m"],
                    loss=info["student.loss2"],
                )
                train_tbx(info, step.sample_num)

                if step.samples_til_eval <= 0:
                    # Evaluate and save checkpoint
                    log.info(f"Evaluating at sample step {step.sample_num}...")
                    # TODO: log all val losses, include MLM
                    loss, val_scores, preds = evaluate(
                        student.model, val_task_loaders, task_datasets, device, args
                    )
                    overall = scorer.scores_to_overall(val_scores)
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
                    metrics = scorer.scores_to_metrics(val_scores)
                    for k in metrics:
                        metrics[k] *= 100
                    metrics.update(overall)

                    val_tbx(metrics, step.sample_num)
                    pseudo_tbx(step.sample_num)
                    pseudo_tbx.clear()


def train_step(x_u, x_m, y_m, x_s, y_s, x_q, y_q, student, teacher, args, step):
    info = {}
    batch_size = x_u.size(0)
    info["batch_size"] = batch_size

    # sample x_hat, y_hat, compute teacher_grad
    teacher_grad, x_hat, y_hat = sample_step(teacher, x_u, args, info)

    # Student x_hat, y_hat forward, backward pass
    deltas = pseudo_step(student, x_hat, y_hat, args, info)

    # Student x_l, y_l forward, backward pass, compute h
    orig_param = support_step(student, x_s, y_s, args, info)
    h = query_step(student, x_q, y_q, orig_param, deltas, args, info)

    # Update teacher grad
    h_step(teacher, teacher_grad, h)

    # MLM
    mlm_step(teacher, x_u, x_m, y_m, args, info)

    # Step teacher
    update_step(teacher, step, batch_size, args, info)

    info["student.lr"] = student.optimizer.param_groups[0]["lr"]
    info["teacher.lr"] = teacher.optimizer.param_groups[0]["lr"]

    return info


def evaluate(model, val_loaders, task_datasets, device, args):
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
                with amp.autocast(enabled=args.autocast):
                    scores = model(x, padding_mask=mask)
                    loss = model.get_loss(scores, y, mask)
                loss_meter.add(loss.item(), batch_size)
                pred = task_dataset.predict(idxs, x, scores)
                preds[name].update(pred)
            val_scores[name] = task_dataset.score(preds[name])

    model.train()
    return loss_meter.avg, val_scores, preds


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
        "--inner_lr", type=float, default=0.1, help="Inner learning rate for student."
    )
    parser.add_argument(
        "--samples_per_task",
        type=int,
        default=4,
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
