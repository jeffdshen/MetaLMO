import numpy as np

import sklearn.metrics

from .record_evaluation import (
    metric_max_over_ground_truths,
    exact_match_score,
    f1_score,
)


def metric_accuracy(preds, examples, strict):
    scores = []
    for example in examples:
        idx = example["idx"]
        if not strict and idx not in preds:
            continue
        scores.append(example["label"] == preds[idx])
    return np.mean(scores)


def metric_f1(preds, examples, strict):
    y_true = []
    y_pred = []
    for example in examples:
        idx = example["idx"]
        if not strict and idx not in preds:
            continue
        y_true.append(example["label"])
        y_pred.append(preds[idx])
    return sklearn.metrics.f1_score(y_true, y_pred, average="macro")


def metric_record(preds, examples, strict):
    f1 = exact_match = total = 0
    for example in examples:
        for qa in example["qas"]:
            if not strict and qa["idx"] not in preds:
                continue

            total += 1

            ground_truths = list(map(lambda x: x["text"], qa["answers"]))
            pred = preds[qa["idx"]]

            exact_match += metric_max_over_ground_truths(
                exact_match_score, pred, ground_truths
            )
            f1 += metric_max_over_ground_truths(f1_score, pred, ground_truths)

    exact_match = exact_match / total
    f1 = f1 / total

    return f1, exact_match


def metric_multirc(preds, examples, strict):
    ems = []
    y_true, y_pred = [], []
    for example in examples:
        passage = example["passage"]
        questions = passage["questions"]
        for question in questions:
            answers = question["answers"]
            labels = [answer["label"] for answer in answers]
            pred_labels = [
                preds[a["idx"]] if strict or a["idx"] in preds else None
                for a in answers
            ]

            zipped_labels = [
                (l, p) for l, p in zip(labels, pred_labels) if p is not None
            ]
            if zipped_labels:
                labels, pred_labels = zip(*zipped_labels)
                em = int(labels == pred_labels)
                ems.append(em)
                y_true += labels
                y_pred += pred_labels

    f1a = sklearn.metrics.f1_score(y_true, y_pred)
    return f1a, np.mean(ems)
