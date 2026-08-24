#!/bin/bash
cd /home/samuel-renteria/Desktop/FILES/PROJECTS/trustfake/wp1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
run() { PYTHONPATH=. .venv/bin/python -m src.run_attacks "$@" || echo "### FAILED: $*"; }

echo "### D2: under-confidence (fast) on base probe + 3 arms"
run --model probe:tf_efficientnet_b0.ns_jft_in1k --split test --attacks underconf --eps 0.0157 --batch 32
for CK in standard_eps2_255 at_pgd_eps2_255 trades_eps2_255; do
  run --model "ckpt:$CK.pt" --split test --attacks underconf --eps 0.0157 --batch 32 --manifest runs/manifest_v2.parquet
done

echo "### D2: AutoAttack (safe composition) on base probe + 3 arms"
run --model probe:tf_efficientnet_b0.ns_jft_in1k --split test --attacks autoattack --eps 0.00784 --batch 32
for CK in standard_eps2_255 at_pgd_eps2_255 trades_eps2_255; do
  run --model "ckpt:$CK.pt" --split test --attacks autoattack --eps 0.00784 --batch 32 --manifest runs/manifest_v2.parquet
done

echo "### D1: ResNet-18 parity — the deck's named baseline"
run --model probe:resnet18.tv_in1k --split calib --attacks clean --batch 32
run --model probe:resnet18.tv_in1k --split test --attacks clean --conditions jpeg_q50 downscale_0.5 --batch 32
run --model probe:resnet18.tv_in1k --split test --attacks ace_uint8 --eps 0.005 --batch 32
run --model probe:resnet18.tv_in1k --split test --attacks overconf --eps 0.0157 --batch 32
echo "### ALL DONE"
