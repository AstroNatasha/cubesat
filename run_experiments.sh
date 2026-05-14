#!/bin/bash
# Run all federated CubeSat experiments and regenerate analysis outputs.
# Each experiment takes ~20–60 min on CPU. Run sequentially or use tmux.

set -e
source .venv/bin/activate

echo "=== Constrained experiments ==="
python train_federated.py --config configs/federated_cubesat_constrained_non_iid.yaml
python train_federated.py --config configs/federated_cubesat_sparse_contacts.yaml

echo "=== Budget sweep ==="
python train_federated.py --config configs/federated_budget_25.yaml
python train_federated.py --config configs/federated_budget_50.yaml
python train_federated.py --config configs/federated_budget_100.yaml
python train_federated.py --config configs/federated_budget_250.yaml

echo "=== Regenerating figures and tables ==="
python analysis/generate_plots.py
python analysis/generate_tables.py

echo "Done. Figures → figures/   Tables → results/tables/"
