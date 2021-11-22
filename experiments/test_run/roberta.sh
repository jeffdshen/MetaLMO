#!/bin/bash

# Run a light version of RoBERTa for testing purposes

command=(
    python run.py roberta_pretrain
    --dim=32
    --n_heads=2
    --ff_dim=128
    --n_layers=3
    --max_positions=512
    --prenorm=True
    --name=test_run
    --num_epochs=2
    --metric_names=loss
    --warmup_steps=100
    --lr=0.008
    --epoch_size=64
    --val_size=16
    --log_per_n_samples=16
    --eval_per_n_samples=32
)
echo ${command[@]}
${command[@]}

