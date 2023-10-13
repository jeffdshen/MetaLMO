# Meta-learning a Language Modeling Objective

## Overview

This repository contains the code for my [Stanford CS330: Deep Multi-Task and Meta Learning](https://cs330.stanford.edu/) final project in Fall 2021 titled "Meta-learning a Language Modeling Objective." The final report can be found [here](./CS330__Project_Report.pdf).

This was an ambitious project to see if we could meta-learn a language modeling objective that improves NLP pretraining. The approach is most similar to [Meta Pseudo Labels](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels). A teacher learns to take unlabeled data, and generate a full input-output text example, which a student pretrains on. The teacher's goal is to improve the student's performance on various downstream tasks.

The paper contains some interesting mathematics extending the gradient update equations for the teacher to the NLP case where inputs and outputs can be full text sequences, as opposed to image data and a label. The code contains a full implementation of MLM (Masked Language Modeling) pretraining [here](./trainers/roberta_pretrainer.py), meta-learning with hard labels [here](./trainers/meta_pretrainer.py), and meta-learning with soft labels [here](./trainers/soft_pretrainer.py). The from-scratch implementation of a pre-norm RoBERTa model comes from my [previous project](https://github.com/jeffdshen/squad).

We performed experiments on a toy Two Moons task and on Wikipedia with SuperGLUE as the downstream tasks. All 8 SuperGLUE tasks and MLM were unified into a single common format for use with Transformer encoder models, e.g. RoBERTa. The code contains implementations for all 8 SuperGLUE tasks and the MLM task [here](./data/tasks.py).

None of the meta-learning approaches performed better than MLM, due to the teacher overfitting to teaching the downstream tasks rather than a general pretraining task. One fix was to force the teacher to only produce the input of a pretraining example, not the full input-output pair. Preliminary experiments showed promise, but due to time constraints, the approach was never fully explored.
