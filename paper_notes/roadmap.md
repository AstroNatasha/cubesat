# Paper Roadmap

## Phase 1 — Baselines (current)
- [x] Folder structure + README
- [x] EuroSAT dataloader
- [x] BaselineCNN definition
- [x] Trainer skeleton
- [x] Evaluation utilities (accuracy, F1, latency)
- [ ] Run baseline_cnn end-to-end; log results to results/

## Phase 2 — Teacher & Student
- [ ] ResNet18 teacher training config + run
- [ ] MobileNetV3-Small baseline run (no KD)
- [ ] KDTrainer (Hinton soft-label loss)
- [ ] KD experiment + results

## Phase 3 — Compression
- [ ] Structured pruning (L1 filter pruning, `torch.nn.utils.prune`)
- [ ] Pruning sensitivity analysis
- [ ] INT8 post-training quantization (`torch.quantization`)
- [ ] Latency benchmarking on CPU

## Phase 4 — Analysis & Paper
- [ ] Pareto plot: accuracy vs. model size
- [ ] Pareto plot: accuracy vs. latency
- [ ] Per-class F1 heat-map across models
- [ ] LaTeX table generation from results CSV

## Open Questions
- Should we also try QAT (quantization-aware training) vs. PTQ?
- Include spectral / multispectral (13-band) EuroSAT variant?
- Target hardware: Raspberry Pi CM4 measurements?
