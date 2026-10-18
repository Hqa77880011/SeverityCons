# SeverityCons

Implementation of **SeverityCons: Severity-Calibrated Anchored Consistency Learning for Corruption-Robust Image Classification**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

`prepare` downloads [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). Download [CIFAR-10-C](https://zenodo.org/records/2535967) separately and extract `labels.npy` and corruption arrays under `data/CIFAR-10-C` (config key `corruption_root`). For CIFAR-100, use `configs/cifar100.json` and [CIFAR-100-C](https://zenodo.org/records/3555552). Corruption benchmark data is evaluation-only.

```bash
python -m severitycons prepare --config configs/cifar10.json
```

## Training and execution

`configs/cifar10.json` fixes seed `2026`. Use `--seed` for a new run and keep its data/config paired. Paths are configurable with the CLI or JSON.

```bash
python -m severitycons train --config configs/cifar10.json --output outputs/severitycons
```

## Evaluation

```bash
python -m severitycons evaluate --config configs/cifar10.json --checkpoint outputs/severitycons/model.pt --output outputs/severitycons
```

## Comparisons and ablations

`compare` runs the configured variants; it includes model fitting where required. Keep comparison outputs separate from the main run.

```bash
python -m severitycons compare --config configs/cifar10.json --output outputs/comparison
python -m severitycons plot --config configs/cifar10.json --output outputs/comparison
```

## Outputs

Run directories contain `model.pt`, `history.csv`, `metrics.json` as produced by the corresponding steps. `config.json` records effective settings. Comparison results are in `outputs/comparison/comparison.csv`; plotting reads those computed files.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
