#!/bin/bash

# SuperGLUE
mkdir -p save/data
wget -q https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip -O save/data/combined.zip
unzip -q save/data/combined.zip -d save/data/
