#!/bin/bash

# Base version run of metalmo
# Uses about 10.871G out of 16.160G

command=(
    python run.py meta_pretrain
    --name=meta_pretrain
    --batch_size=4
    --prenorm=True
    --num_epochs=1000
    --metric_names loss Overall
    --lr=0.01
)
echo ${command[@]}
${command[@]}
