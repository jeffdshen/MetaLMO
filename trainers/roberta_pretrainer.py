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
from trainers.util import (
    cat_pred_examples,
    evaluate,
    pseudo_step,
    query_step,
    real_step,
    score_evaluate,
    support_step,
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
    log_tbx = stats.TensorboardScalars(
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
    val_tasks = ["MLM"]
    val_tbx = stats.TensorboardScalars(
        tbx,
        "val",
        scorer.get_overall_names(val_tasks) + scorer.get_metric_names(val_tasks),
    )
    student_tbx = stats.TensorboardWeights(tbx, "student")
    formatter = stats.TokenizedTextFormatter(pretrain_tokenizer, ["idx", "x", "y", "pred"])
    text_tbxs = stats.TensorboardTexts(
        tbx, "val", "example_{}", val_tasks, formatter, args.num_visuals
    )
    return train_tbx, log_tbx, val_tbx, student_tbx, text_tbxs


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
    (
        meta_dataset,
        meta_loader,
        task_datasets,
        task_loaders,
        mlm_task_datasets,
        mlm_task_loaders,
    ) = config.get_pretrain_datasets(args, task_tokenizer, pretrain_tokenizer)
    scorer = config.get_scorer(args)
    train_tbx, log_tbx, val_tbx, student_tbx, text_tbxs = get_stats(
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

                info = train_step(
                    x_u, x_m, y_m, x_s, y_s, x_q, y_q, student, args, step
                )
                # Log info
                progress_bar.update(1)
                progress_bar.set_postfix(
                    epoch=step.epoch,
                    loss=info["student.loss0"],
                )
                if "h" not in info:
                    train_tbx(info, step.sample_num)
                else:
                    log_tbx(info, step.sample_num)

                if step.samples_til_eval <= 0:
                    # Evaluate and save checkpoint
                    log.info(f"Evaluating at sample step {step.sample_num}...")

                    losses, val_scores, tensors, preds = evaluate(
                        student.model,
                        mlm_task_loaders,
                        mlm_task_datasets,
                        ["MLM"],
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


def train_step(x_u, x_m, y_m, x_s, y_s, x_q, y_q, student, args, step):
    info = {}
    batch_size = x_u.size(0)
    info["batch_size"] = batch_size

    # Student x_m, y_m forward, backward pass
    real_step(student, x_m, y_m, args, info)

    # Step student
    if (
        step.samples_til_log <= 0
        and (step.step_num + 1) % args.gradient_accumulation == 0
    ):
        with torch.no_grad():
            deltas = [param.data.clone() for param in student.model.parameters()]
        update_step(student, step, batch_size, args, info)
        with torch.no_grad():
            for delta, param in zip(deltas, student.model.parameters()):
                delta -= param.data

        # Student x_l, y_l forward, backward pass, compute h
        orig_param = support_step(student, x_s, y_s, args, info)
        _ = query_step(student, x_q, y_q, orig_param, deltas, args, info)

        step.samples_til_log = args.log_per_n_samples
    else:
        update_step(student, step, batch_size, args, info)

    step.samples_til_log -= batch_size

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
