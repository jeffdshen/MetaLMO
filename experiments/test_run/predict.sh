#!/bin/bash

# Run a light version of finetuning for testing purposes

command=(
    python run.py predict
    --name=nlp/predict/COPA/meta
    --dim=32
    --n_heads=2
    --ff_dim=128
    --n_layers=3
    --max_positions=512
    --prenorm=True
    --val_size=16
    --task=COPA
)
echo ${command[@]}
${command[@]}