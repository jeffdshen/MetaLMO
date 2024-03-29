#!/bin/bash

# Two moon version run of metalmo

command=(
    python run.py meta_pretrain
    --name=two_moons/pretrain/meta
    --max_positions 34
    --context_window_stride 0
    --tokenizer_dir=save/tokenizers/two-moons
    --dataset=two_moons
    --batch_size=4
    --samples_per_task=4
    --gradient_accumulation=4
    --prenorm=True
    --dim=4
    --n_heads=2
    --ff_dim=16
    --n_layers=2
    --num_epochs=10
    --epoch_size=32000
    --eval_per_n_samples=4000
    --metric_names loss Overall
    --lr=0.01
    --warmup_steps=100
)
echo ${command[@]}
${command[@]}
