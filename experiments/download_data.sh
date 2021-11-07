#!/bin/bash

# SuperGLUE
mkdir -p save/data
curl --silent https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip -o save/data/combined.zip
unzip -q save/data/combined.zip -d save/data/
rm save/data/combined.zip

