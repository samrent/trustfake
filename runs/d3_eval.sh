#!/bin/bash
# WP3 comparative evaluation: the matched triple over clean / realistic / adversarial.
set -e
cd /home/samuel-renteria/Desktop/FILES/PROJECTS/trustfake/wp1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
for CK in standard_eps2_255 at_pgd_eps2_255 trades_eps2_255; do
  echo "=== $CK ==="
  PYTHONPATH=. .venv/bin/python -m src.run_attacks --model "ckpt:$CK.pt" --split calib \
    --attacks clean --batch 32 --manifest runs/manifest_v2.parquet
  PYTHONPATH=. .venv/bin/python -m src.run_attacks --model "ckpt:$CK.pt" --split test \
    --attacks clean --conditions jpeg_q50 downscale_0.5 --batch 32 --manifest runs/manifest_v2.parquet
  PYTHONPATH=. .venv/bin/python -m src.run_attacks --model "ckpt:$CK.pt" --split test \
    --attacks ace_uint8 --eps 0.005 --batch 32 --manifest runs/manifest_v2.parquet
  PYTHONPATH=. .venv/bin/python -m src.run_attacks --model "ckpt:$CK.pt" --split test \
    --attacks overconf --eps 0.0157 --batch 32 --manifest runs/manifest_v2.parquet
  PYTHONPATH=. .venv/bin/python -m src.run_attacks --model "ckpt:$CK.pt" --split test \
    --attacks pgd_linf --eps 0.00784 --batch 32 --manifest runs/manifest_v2.parquet
done
