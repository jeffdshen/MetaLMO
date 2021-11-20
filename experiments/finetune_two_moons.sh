#!/bin/bash

# Two moon version run of metalmo
if [ "$#" -eq 0 ]; then
  other_args=""
else
  other_args="$@"
fi

command=(
    python run.py finetune
    --name=meta_finetune_two_moons
    --max_positions 34
    --context_window_stride 0
    --tokenizer_dir=save/tokenizers/two-moons
    --dataset=two_moons
    --batch_size=1
    --gradient_accumulation=1
    --prenorm=True
    --dim=32
    --n_heads=2
    --ff_dim=64
    --n_layers=2
    --num_epochs=100
    --eval_per_n_samples=5
    --metric_names loss Overall
    --lr=0.005
    --warmup_steps=10
    --task=Which_MOON
    $other_args
)
echo ${command[@]}
${command[@]}
