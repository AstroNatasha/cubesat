# Resource-Aware Model Compression for Onboard CubeSat Image Classification

Research codebase accompanying the paper:  
**"Resource-Aware Model Compression for Onboard CubeSat Image Classification"**

## Overview

This repository investigates the trade-off between classification accuracy and computational
footprint when deploying deep learning models on CubeSat platforms. We evaluate knowledge
distillation, structured pruning, and INT8 quantization applied to lightweight CNN architectures
trained on the EuroSAT remote sensing dataset.

## Research Questions

1. How much accuracy is lost when compressing a ResNet18 teacher into a MobileNetV3-Small student via knowledge distillation?
2. Can structured pruning reduce parameter count by ≥50% with <2% accuracy drop?
3. Does INT8 post-training quantization preserve F1 score while halving model size?
4. What is the Pareto frontier between latency and accuracy across compression configurations?

## Pipeline

```
EuroSAT dataset
    │
    ├─► Baseline CNN  (from scratch)
    ├─► MobileNetV3-Small  (from scratch / fine-tuned)
    └─► ResNet18 teacher  ──► Knowledge Distillation ──► Student
                                        │
                               Structured Pruning
                                        │
                               INT8 Quantization
                                        │
                              Evaluation & Metrics
```

## Metrics

| Metric | Description |
|---|---|
| Top-1 Accuracy | Per-class and overall |
| Macro F1 | Class-balanced performance |
| Model size | On-disk `.pt` file size (MB) |
| Parameters | Trainable parameter count |
| Latency | Inference time (ms) on CPU / target hardware |

## Project Structure

```
cubesat/
├── configs/            # YAML experiment configs
├── src/
│   ├── data/           # EuroSAT dataset loader and transforms
│   ├── models/         # Baseline CNN, MobileNetV3-Small, ResNet18
│   ├── training/       # Training loops, loss functions, KD trainer
│   ├── compression/    # Pruning and quantization utilities
│   └── evaluation/     # Metrics, latency benchmarking, reporting
├── experiments/        # Per-run output dirs (checkpoints, logs)
├── results/            # Aggregated CSVs and figures for the paper
├── paper_notes/        # LaTeX snippets, experiment notes, todos
├── train.py            # Main training entry point
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# Train baseline CNN
python train.py --config configs/baseline_cnn.yaml

# Train MobileNetV3-Small with knowledge distillation
python train.py --config configs/kd_mobilenet.yaml

# Evaluate and export metrics
python evaluate.py --checkpoint experiments/kd_mobilenet/best.pt
```

## Datasets

**EuroSAT** — Sentinel-2 multispectral satellite imagery, 10 land-use classes, 27,000 images at 64×64px.  
Source: https://github.com/phelber/EuroSAT  
Download is handled automatically via `torchvision.datasets.EuroSAT`.

## Target Hardware Context

CubeSat onboard computers (e.g., Raspberry Pi CM4, Jetson Nano, LEON3 FPGA) impose:
- RAM: 256 MB – 4 GB
- Storage: 4–32 GB
- Power budget: 1–5 W inference

Models are evaluated on CPU to approximate constrained-hardware latency.

## Status

- [ ] Dataset loader (EuroSAT)
- [ ] Baseline CNN
- [ ] MobileNetV3-Small
- [ ] ResNet18 teacher
- [ ] Training loop
- [ ] Knowledge distillation
- [ ] Structured pruning
- [ ] INT8 quantization
- [ ] Evaluation & metrics
- [ ] Results aggregation
