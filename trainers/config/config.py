# Copyright (c) Jeffrey Shen

import json
import pathlib

import numpy as np
import torch
import torch.optim as optim

import models
import trainers.schedulers as schedulers
import trainers.stats as stats
from trainers.state import ModelSaver

from .data import get_dataset_maximize_metric


def bool_arg(s):
    return (s.lower().startswith("t"),)


def get_maximize_metric():
    maximize_metric = {
        "loss": False,
        "loss0": False,
        "loss2": False,
    }
    maximize_metric.update(get_dataset_maximize_metric())
    return maximize_metric


def get_roberta_model(args, max_tokens):
    model = models.RoBERTa(
        dim=args.dim,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        activation=args.activation,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        act_dropout=args.act_dropout,
        n_layers=args.n_layers,
        max_positions=args.max_positions,
        max_tokens=max_tokens,
        padding_idx=args.padding_idx,
        ignore_idx=args.ignore_idx,
        prenorm=args.prenorm,
    )
    return model


def add_roberta_args(parser):
    parser.add_argument(
        "--dim",
        type=int,
        default=768,
        help="Embedding dimension.",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=12,
        help="Attention heads.",
    )
    parser.add_argument(
        "--ff_dim",
        type=int,
        default=3072,
        help="Feedforward dimension.",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "gelu"],
        default="gelu",
        help="Feedforward activation function.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--attn_dropout",
        type=float,
        default=0.1,
        help="Dropout probability for attention weights within self attn.",
    )
    parser.add_argument(
        "--act_dropout",
        type=float,
        default=0.0,
        help="Dropout probability after activation within FF.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=12,
        help="Number of layers.",
    )
    parser.add_argument(
        "--prenorm",
        type=bool_arg,
        default=False,
        help="Whether to put LayerNorm after the residual or before.",
    )


def get_teacher_model(args, max_tokens):
    # TODO Read from a config instead. Currently just shares student args
    model = models.TeacherRoBERTa(
        dim=args.dim,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        activation=args.activation,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        act_dropout=args.act_dropout,
        n_layers=args.n_layers,
        max_positions=args.max_positions,
        max_tokens=max_tokens,
        padding_idx=args.padding_idx,
        ignore_idx=args.ignore_idx,
        prenorm=args.prenorm,
    )
    return model


def get_available_devices():
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f"cuda:{gpu_ids[0]}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return device, gpu_ids


def update_gpus(args):
    device, args.gpu_ids = get_available_devices()

    args.batch_size_per_gpu = args.batch_size
    args.batch_size *= max(1, len(args.gpu_ids))

    return device


def get_adamw_optimizer(args, model):
    return optim.AdamW(
        model.parameters(),
        args.lr,
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
        weight_decay=args.l2_wd,
    )


def add_adamw_optimizer_args(parser):
    parser.add_argument("--lr", type=float, default=0.08, help="Learning rate.")
    parser.add_argument("--l2_wd", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Adam epsilon.")
    parser.add_argument("--beta_1", type=float, default=0.9, help="Adam beta_1.")
    parser.add_argument("--beta_2", type=float, default=0.98, help="Adam beta_2.")


def get_lwpd_scheduler(args, optimizer, total_steps):
    if args.decay_forever:
        total_steps = float("inf")
    return schedulers.get_linear_warmup_power_decay_scheduler(
        optimizer, args.warmup_steps, total_steps, power=args.power_decay
    )


def add_lwpd_scheduler_args(parser):
    parser.add_argument(
        "--warmup_steps", type=float, default=10000, help="Warmup optimizer steps."
    )
    parser.add_argument(
        "--power_decay", type=float, default=-0.5, help="Power of the decay."
    )
    parser.add_argument(
        "--decay_forever",
        type=lambda s: s.lower().startswith("t"),
        default=True,
        help="Whether the decay should reach end_lr at the end of training, or in the limit to infinity",
    )


def get_model_saver(args, log):
    return ModelSaver(
        args.save_dir,
        metric_names=args.metric_names,
        maximize_metric=args.maximize_metric,
        log=log,
    )


def add_model_saver_args(parser, metric_names, maximize_metric=get_maximize_metric()):
    parser.add_argument(
        "--metric_names",
        type=str,
        nargs="*",
        default=metric_names,
        choices=metric_names,
        help="Name(s) of dev metrics to determine best checkpoint.",
    )
    parser.set_defaults(maximize_metric=maximize_metric)


def add_train_test_args(parser):
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        required=True,
        help="Name to identify training or test run.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for all random generators."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./save/runs/",
        help="Base directory for saving runs.",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to load as a model checkpoint.",
    )


def add_train_args(parser):
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Path to trainer checkpoint.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Number of epochs for which to train. Negative means forever.",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=1,
        help="Number of forward passes for each backward pass",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="Maximum gradient norm for gradient clipping.",
    )


def get_save_dir(base_dir, sub_dir, name, id_max=100):
    for id in range(1, id_max):
        save_dir: pathlib.Path = pathlib.Path(base_dir) / sub_dir / f"{name}-{id:02d}"
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=False)
            return str(save_dir)
    raise RuntimeError("Too many save directories with the same name")


def save_checkpoint(args, state):
    checkpoint_path = pathlib.Path(args.save_dir, "checkpoint.pth.tar")
    checkpoint = state.state_dict()
    torch.save(checkpoint, checkpoint_path)


def reload_checkpoint(args, state, log=None):
    if args.resume_dir:
        checkpoint_path = pathlib.Path(args.resume_dir, "checkpoint.pth.tar")
        state.load_state_dict(torch.load(checkpoint_path))
        if log is not None:
            log.info("Resuming from checkpoint: {}".format(checkpoint_path))
