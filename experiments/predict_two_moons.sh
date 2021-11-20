#!/bin/bash

# Predict on the two moon dataset
if [ "$#" -eq 0 ]; then
  other_args=""
else
  other_args="$@"
fi

command=(
    python run.py predict
    --name=meta_predict_two_moons
    --max_positions 34
    --context_window_stride 0
    --tokenizer_dir=save/tokenizers/two-moons
    --dataset=two_moons
    --batch_size=4
    --prenorm=True
    --dim=32
    --n_heads=2
    --ff_dim=64
    --n_layers=2
    --task=Which_MOON
    $other_args
)
echo ${command[@]}
${command[@]}