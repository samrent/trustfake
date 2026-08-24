#!/bin/bash
cd /home/samuel-renteria/Desktop/FILES/PROJECTS/trustfake/wp1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
f(){ PYTHONPATH=. .venv/bin/python "$@" || echo "### FAILED: $*"; }
echo "### 1. extract ResNet-18 features (the missing step)"
f -m src.features --model resnet18.tv_in1k --splits fit calib test --batch 64
echo "### 2. ResNet-18 attacks"
f -m src.run_attacks --model probe:resnet18.tv_in1k --split calib --attacks clean --batch 32
f -m src.run_attacks --model probe:resnet18.tv_in1k --split test --attacks clean --conditions jpeg_q50 downscale_0.5 --batch 32
f -m src.run_attacks --model probe:resnet18.tv_in1k --split test --attacks ace_uint8 --eps 0.005 --batch 32
f -m src.run_attacks --model probe:resnet18.tv_in1k --split test --attacks overconf --eps 0.0157 --batch 32
echo "### RESNET DONE"
