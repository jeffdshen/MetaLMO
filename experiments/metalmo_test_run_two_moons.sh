#!/bin/bash

# Run a light version of MetaLMO on TWO_MOONS

command=(
    python run.py meta_pretrain
    --name=meta_pretrain_two_moons
    --max_positions 34
    --context_window_stride 0
    --tokenizer_dir=save/tokenizers/two-moons
    --dataset=two_moons
    --batch_size=4
    --samples_per_task=4
    --prenorm=True
    --dim=32
    --n_heads=2
    --ff_dim=64
    --n_layers=2
    --num_epochs=2
    --epoch_size=32
    --eval_per_n_samples=16
    --metric_names loss Overall
    --lr=0.01
    --warmup_steps=100
)
echo ${command[@]}
${command[@]}
