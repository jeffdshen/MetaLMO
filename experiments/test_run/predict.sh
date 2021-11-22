#!/bin/bash

# Run a light version of finetuning for testing purposes

command=(
    python run.py predict
    --dim=32
    --n_heads=2
    --ff_dim=128
    --n_layers=3
    --max_positions=512
    --prenorm=True
    --name=test_run
    --val_size=16
    --task=COPA
)
echo ${command[@]}
${command[@]}