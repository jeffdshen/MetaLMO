#!/bin/bash

# Run a light version of finetuning for testing purposes

command=(
    python run.py finetune
    --name=nlp/finetune/COPA/meta
    --dim=32
    --n_heads=2
    --ff_dim=128
    --n_layers=3
    --max_positions=512
    --prenorm=True
    --num_epochs=2
    --metric_names=loss
    --warmup_steps=100
    --lr=0.004
    --val_size=16
    --eval_per_n_samples=32
    --task=COPA
)
echo ${command[@]}
${command[@]}
