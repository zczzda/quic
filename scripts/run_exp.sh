#!/usr/bin/env bash
set -euo pipefail

python train.py --epochs 2
python eval.py --checkpoint outputs/best.pt
