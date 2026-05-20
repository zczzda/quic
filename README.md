# HFS-Net Minimal Experiment Scaffold

This repository contains a minimal runnable PyTorch implementation for HFS-Net style experiments:
- Sequence branch (Traffic-Mamba style lightweight SSM surrogate)
- Statistical feature branch (MLP)
- Cross-space dynamic gating fusion (CSDG)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --epochs 2
python eval.py --checkpoint outputs/best.pt
```

> Notes: For portability, this scaffold uses a lightweight gated state-space block implemented with standard PyTorch operators, without requiring external Mamba CUDA kernels.
