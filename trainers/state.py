# Copyright (c) Jeffrey Shen

import os
import random
import math
from dataclasses import dataclass

import numpy as np
import torch


class TrainerState:
    def __init__(self, log=None):
        super().__init__()
        self.state = {}
        self.objs = {}
        self.log = log

    def load_state_dict(self, state_dict):
        self.state = state_dict

    def is_reloading(self):
        return bool(self.state)

    def track_object(self, name, obj):
        self.objs[name] = obj
        if name in self.state:
            if self.log is not None:
                self.log.info("Reloading {}".format(name))
            obj.load_state_dict(self.state.pop(name))
            return True
        return False

    def state_dict(self):
        state = {}
        for name, obj in self.objs.items():
            state[name] = obj.state_dict()
        return state


class KeepSimpleState:
    def __init__(self, keep, **kwargs):
        self.__dict__.update(kwargs)
        self.__keep = keep

    @staticmethod
    def init_from(other, keep):
        return KeepSimpleState(keep, **other.__dict__)

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state):
        keep = self.__keep
        keep_dict = {k: self.__dict__[k] for k in self.__keep}
        self.__dict__ = state
        self.__dict__.update(keep_dict)
        self.__keep = keep


class SimpleState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def init_from(other):
        return SimpleState(**other.__dict__)

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state):
        self.__dict__ = state


class RandomState:
    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def state_dict(self):
        return {
            "random": random.getstate(),
            "np.random": np.random.get_state(),
            "torch.random": torch.random.get_rng_state(),
            "torch.cuda.random": torch.cuda.get_rng_state_all(),
        }

    def load_state_dict(self, state):
        random.setstate(state["random"])
        np.random.set_state(state["np.random"])
        torch.set_rng_state(state["torch.random"])
        torch.cuda.set_rng_state_all(state["torch.cuda.random"])


@dataclass
class ModelState:
    model = None
    optimizer = None
    scheduler = None
    scaler = None
    noop_optimizer = None


class ModelSaver:
    def __init__(self, save_dir, metric_names, maximize_metric, log):
        super().__init__()

        self.save_dir = save_dir
        self.metric_names = metric_names
        self.maximize_metric = maximize_metric
        self.best_vals = {}
        self.log = log
        maximize = ", ".join(m for m in metric_names if maximize_metric[m])
        minimize = ", ".join(m for m in metric_names if not maximize_metric[m])
        if maximize:
            self.log.info(f"Best model saver will maximize {maximize}")
        if minimize:
            self.log.info(f"Best model saver will minimize {minimize}")

    def save(self, step, model, results):
        ckpt_dict = {
            "model": model.state_dict(),
            "step": step,
        }

        for metric in self.metric_names:
            if metric not in results:
                continue
            value = results[metric]
            maximize = self.maximize_metric[metric]
            if math.isnan(value):
                continue
            if metric in self.best_vals:
                best_val = self.best_vals[metric]
                if maximize and best_val >= value:
                    continue
                if not maximize and best_val <= value:
                    continue

            # value is better than best_val
            self.best_vals[metric] = value
            best_path = os.path.join(self.save_dir, f"best_{metric}.pth.tar")
            self.log.info(f"New best checkpoint for metric {metric} at step {step}...")
            torch.save(ckpt_dict, best_path)

    @staticmethod
    def load_model(model, checkpoint_path, device, strict):
        ckpt_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt_dict["model"], strict=strict)
        step = ckpt_dict["step"]
        return model, step
