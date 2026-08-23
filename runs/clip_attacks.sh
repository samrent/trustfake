#!/bin/bash
cd /home/samuel-renteria/Desktop/FILES/PROJECTS/trustfake/wp1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
M="--model probe:vit_large_patch14_clip_224.openai --model-id vit_l14_e2e --batch 16 --sample 4000"
echo "### clean"
PYTHONPATH=. .venv/bin/python -m src.run_attacks $M --split test --attacks clean || echo "### FAILED clean"
for A in "ace_uint8 0.005" "ace 0.005" "pgd_linf 0.00784" "overconf 0.0157"; do
  set -- $A; echo "### $1 eps=$2"
  PYTHONPATH=. .venv/bin/python -m src.run_attacks $M --split test --attacks "$1" --eps "$2" || echo "### FAILED $1"
done
echo "### ALL DONE"
