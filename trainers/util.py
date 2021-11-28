# Copyright (c) Jeffrey Shen

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.cuda.amp as amp
from trainers.optimizers import GradSaver

from trainers.state import (
    ModelState,
)
import trainers.stats as stats
import models.transformer as T


def soft_sample_step(teacher: ModelState, x_u, args, info):
    mask_x_u = T.get_padding_mask(x_u, args.padding_idx)
    with amp.autocast(enabled=False):
        scores_x, scores_y = teacher.model(x_u, padding_mask=mask_x_u)
        x_hat, y_hat = teacher.model.sample(
            scores_x, scores_y, x_u, x_u, mask_x_u, mask_x_u, k=1
        )
        x_hat, y_hat = x_hat.squeeze(0), y_hat.squeeze(0)

    info["pseudo"] = stats.tensors_to_lists((x_u, x_hat, y_hat))
    return scores_x, scores_y, mask_x_u


def soft_pseudo_step(
    diff_student: ModelState, scores_x, scores_y, mask_x_u, args, info
):
    # NOTE: No autocast, because GradScaler is non-differentiable, and is very
    # annoying to replace.
    with amp.autocast(enabled=False):
        scores = diff_student.model(scores_x, padding_mask=mask_x_u)
        loss = diff_student.model.get_loss(scores, scores_y, mask_x_u)
    info["student.loss0"] = loss.item()

    # Save grad to add to student
    grad_saver = GradSaver()
    diff_student.optimizer.step(
        loss / args.gradient_accumulation, grad_callback=grad_saver
    )
    return grad_saver


def soft_support_step(diff_student: ModelState, x_s, y_s, args, info):
    mask_x_s = T.get_padding_mask(x_s, args.padding_idx)
    with amp.autocast(enabled=False):
        scores = diff_student.model(x_s, padding_mask=mask_x_s)
        loss = diff_student.model.get_loss(scores, y_s, mask_x_s)
    info["student.loss1"] = loss.item()
    diff_student.optimizer.step(loss)


def soft_query_step(diff_student: ModelState, x_q, y_q, args, info):
    mask_x_q = T.get_padding_mask(x_q, args.padding_idx)
    with amp.autocast(enabled=False):
        scores = diff_student.model(x_q, padding_mask=mask_x_q)
        loss = diff_student.model.get_loss(scores, y_q, mask_x_q)
    info["student.loss2"] = loss.item()
    return loss


def sample_step(teacher: ModelState, x_u, args, info):
    mask_x_u = T.get_padding_mask(x_u, args.padding_idx)
    with amp.autocast(enabled=args.autocast):
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
    info["pseudo"] = stats.tensors_to_lists((x_u, x_hat, y_hat))
    return teacher_grad, x_hat, y_hat


def pseudo_step(student: ModelState, x_hat, y_hat, args, info):
    with torch.no_grad():
        deltas = [param.data.detach().clone() for param in student.model.parameters()]
    mask_x_hat = T.get_padding_mask(x_hat, args.padding_idx)
    with amp.autocast(enabled=args.autocast):
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
    with torch.no_grad():
        for delta, param in zip(deltas, student.model.parameters()):
            delta -= param.data
    return deltas


def real_step(student: ModelState, x, y, args, info):
    mask_x = T.get_padding_mask(x, args.padding_idx)
    mask_y = T.get_padding_mask(y, args.padding_idx)
    with amp.autocast(enabled=args.autocast):
        scores = student.model(x, padding_mask=mask_x)
        loss = student.model.get_loss(scores, y, mask_y)
    info["student.loss0"] = loss.item()
    student.scaler.scale(loss / args.gradient_accumulation).backward()


def support_step(student: ModelState, x_s, y_s, args, info):
    with torch.no_grad():
        orig_param = [
            param.data.detach().clone() for param in student.model.parameters()
        ]
    mask_x_s = T.get_padding_mask(x_s, args.padding_idx)
    with amp.autocast(enabled=args.autocast):
        scores = student.model(x_s, padding_mask=mask_x_s)
        loss = student.model.get_loss(scores, y_s, mask_x_s)
    info["student.loss1"] = loss.item()
    student.scaler.scale(loss).backward()
    student.scaler.unscale_(student.noop_optimizer)
    nn.utils.clip_grad_norm_(student.model.parameters(), args.max_grad_norm)
    with torch.no_grad():
        for param in student.model.parameters():
            param.data.add_(param.grad, alpha=-args.inner_lr)
    student.scaler.step(student.noop_optimizer)
    student.scaler.update()
    student.noop_optimizer.zero_grad()
    return orig_param


def query_step(student: ModelState, x_q, y_q, orig_param, deltas, args, info):
    mask_x_q = T.get_padding_mask(x_q, args.padding_idx)
    with amp.autocast(enabled=args.autocast):
        scores = student.model(x_q, padding_mask=mask_x_q)
        loss = student.model.get_loss(scores, y_q, mask_x_q)
    info["student.loss2"] = loss.item()
    student.scaler.scale(loss).backward()
    student.scaler.unscale_(student.noop_optimizer)
    nn.utils.clip_grad_norm_(student.model.parameters(), args.max_grad_norm)
    with torch.no_grad():
        for orig, param in zip(orig_param, student.model.parameters()):
            param.data = orig.data
    student.scaler.step(student.noop_optimizer)
    student.scaler.update()
    h = calc_h_step(student, deltas, info)
    student.noop_optimizer.zero_grad()
    return h


def calc_h_step(student: ModelState, deltas, info):
    with torch.no_grad():
        h = 0.0
        delta_norm = 0.0
        grad_norm = 0.0
        for delta, param in zip(deltas, student.model.parameters()):
            h += delta.flatten().dot(param.grad.flatten())
            delta_norm += (delta ** 2).sum()
            grad_norm += (param.grad ** 2).sum()
        info["unscaled_h"] = h.item()
        h = h / (torch.sqrt(delta_norm) * torch.sqrt(grad_norm))
    info["h"] = h.item()
    return h


def h_step(teacher: ModelState, teacher_grad, h):
    # nan or inf or -inf
    if not (-1 <= h <= 1):
        return

    for grad, param in zip(teacher_grad, teacher.model.parameters()):
        if param.grad is None:
            param.grad = h * grad
        else:
            param.grad += h * grad


def mlm_step(teacher: ModelState, x_u, x_m, y_m, args, info):
    # Train teacher to produce MLM
    mask_x_u = T.get_padding_mask(x_u, args.padding_idx)
    x_u[:, 0] = x_m[:, 0]
    # NOTE: T(x_m, y_m | x_u) won't work because they are not independent,
    # so we learn T(x_u, x_m | x_u)
    with amp.autocast(enabled=args.autocast):
        scores_x, scores_y = teacher.model(x_u, padding_mask=mask_x_u)
        loss = teacher.model.get_loss(scores_x, scores_y, x_m, x_u, mask_x_u, mask_x_u)
    info["teacher.loss_x_m"] = loss.item()
    teacher.scaler.scale(loss / args.gradient_accumulation).backward()

    # Train teacher with MLM
    mask_y_m = T.get_padding_mask(y_m, args.padding_idx)
    with amp.autocast(enabled=args.autocast):
        scores_x, scores_y = teacher.model(x_m, padding_mask=mask_x_u)
        loss = teacher.model.get_loss(scores_x, scores_y, x_m, y_m, mask_x_u, mask_y_m)
    info["teacher.loss_y_m"] = loss.item()
    teacher.scaler.scale(loss / args.gradient_accumulation).backward()


def optim_step(state: ModelState, step, args):
    if (step.step_num + 1) % args.gradient_accumulation == 0:
        state.scaler.unscale_(state.optimizer)
        nn.utils.clip_grad_norm_(state.model.parameters(), args.max_grad_norm)
        state.scaler.step(state.optimizer)
        state.scaler.update()
        state.scheduler.step()
        state.optimizer.zero_grad()


def update_step(step, batch_size, args, info):
    step.step_num += 1
    step.sample_num += batch_size
    if step.samples_til_eval <= 0:
        step.samples_til_eval = args.eval_per_n_samples
    step.samples_til_eval -= batch_size
    info["steps"] = step.step_num / args.gradient_accumulation


def evaluate(model, loaders, datasets, names, split, device, args):
    model.eval()
    loss_meters = {}
    tensors = {}
    preds = {}
    val_scores = {}
    with torch.no_grad():
        for name in names:
            preds[name] = {}
            tensors[name] = {}
            loss_meters[name] = stats.AverageMeter()
            single_scores = {}
            for idxs, x, y in loaders[name][split]:
                batch_size = x.size(0)
                x, y = x.to(device), y.to(device)
                mask_x = T.get_padding_mask(x, args.padding_idx)
                mask_y = T.get_padding_mask(y, args.padding_idx)
                with amp.autocast(enabled=args.autocast):
                    scores = model(x, padding_mask=mask_x)
                    loss = model.get_loss(scores, y, mask_y)
                loss_meters[name].add(loss.item(), batch_size)
                pred = datasets[name][split].predict(idxs, x, scores)
                preds[name].update(pred)
                tensors[name].update(stats.tensors_groupby_flatten(idxs, [x, y]))

                single_score = datasets[name][split].score_single(idxs, pred, y)
                if single_score is not None:
                    single_scores.update(single_score)
            if single_scores:
                val_scores[name] = np.mean(list(single_scores.values()))
            else:
                val_scores[name] = datasets[name][split].score(preds[name])

    model.train()
    losses = {name: loss_meter.avg for name, loss_meter in loss_meters.items()}

    return losses, val_scores, tensors, preds


def cat_pred_examples(tensors, preds):
    return {
        name: [
            [idx] + list(tensors[name][idx]) + [preds[name][idx]]
            for idx in tensors[name]
        ]
        for name in tensors
    }


def score_evaluate(scorer, val_scores, losses):
    val_scores = val_scores.copy()
    scorer.scale_scores(val_scores, 100)
    val_scores.update({k + "_loss": v for k, v in losses.items()})
    overall = scorer.scores_to_overall(val_scores)

    overall_str = ", ".join(f"{k}: {v:05.2f}" for k, v in overall.items())

    metrics = scorer.scores_to_metrics(val_scores)
    metrics.update(overall)
    return overall, overall_str, metrics
