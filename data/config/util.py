import numpy as np


def add_all_to_overall(overall, scores, names):
    for metric in names:
        if metric in scores:
            overall[metric] = scores[metric]


def add_mean_to_overall(overall, scores, names, overall_name):
    if any(name in scores for name in names):
        overall[overall_name] = np.mean(
            [scores[name] for name in names if name in scores]
        )


def add_mean_of_mean_to_overall(overall, scores, names, overall_name):
    if any(name in scores for name in names):
        overall[overall_name] = np.mean(
            [np.mean(scores[name]) for name in names if name in scores]
        )


def add_suffix(names, suffix):
    return [name + suffix for name in names]
