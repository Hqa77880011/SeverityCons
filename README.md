# SeverityCons

Implementation of **SeverityCons: Severity-Calibrated Anchored Consistency Learning for Corruption-Robust Image Classification**. Training pairs a clean image with one corrupted view and weights clean-anchored prediction and feature consistency by corruption severity.

## Installation

Use Python 3.10 or newer. Install matching PyTorch and torchvision builds for your device, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. For NVIDIA GPUs, choose a CUDA build from the [PyTorch installer](https://pytorch.org/get-started/locally/) in place of the CPU command. The supported dependency range is PyTorch 2.4–2.7 with its matching torchvision release.

## Data and workflow

`prepare` downloads [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) through torchvision and creates a seeded, class-balanced validation split of 5,000 training images. The remaining 45,000 images form the training pool; the official test split is untouched. The prepared directory contains `train.npz`, `val.npz`, `test.npz` and `meta.json`. Array indices identify the original training examples. A custom image collection can use `dataset="folder"` with matching class directories under `root/train`, `root/val` and `root/test`.

Download and extract [CIFAR-10-C](https://zenodo.org/records/2535967) or [CIFAR-100-C](https://zenodo.org/records/3555552). Set `corruption_root` to the directory containing `labels.npy` and the corruption arrays. Each corruption array has five consecutive severity blocks. These arrays are memory mapped during evaluation. They are excluded from training and validation.

```bash
python -m severitycons prepare --config configs/cifar10.json
python -m severitycons train --config configs/cifar10.json --output outputs/severitycons
python -m severitycons evaluate --config configs/cifar10.json --checkpoint outputs/severitycons/model.pt --output outputs/severitycons
python -m severitycons plot --config configs/cifar10.json --output outputs/severitycons
```

`configs/cifar100.json` supplies the CIFAR-100 paths. If `corruption_root` is an empty string, evaluation measures only clean test accuracy. Otherwise all selected corruption files must be present.

## Baselines and ablations

```bash
python -m severitycons compare --config configs/cifar10.json --output outputs/comparison
python -m severitycons plot --config configs/cifar10.json --output outputs/comparison
```

| Variant | Difference |
| --- | --- |
| `clean` | Clean supervised training |
| `fixed` | Unit consistency weights |
| `increasing` | `exp(eta*s)` severity weights |
| `decreasing` | `exp(-eta*s)` without a floor |
| `severitycons` | `max(floor, exp(-eta*s))` |
| `no_prediction` | Removes prediction consistency |
| `no_feature` | Removes feature consistency |
| `no_anchor` | Removes stop-gradient on clean anchors |

All consistency variants receive the same random views, training budget and seeds. `compare` trains each entry in `variants` for every entry in `seeds`.

## Configuration and outputs

Commands read the supplied JSON configuration. `--data`, `--output`, `--device`, `--seed`, `--batch-size`, `--epochs`, `--max-steps` and `--limit` override individual fields. `--device auto` chooses CUDA when available. `max_steps=0` and `limit=0` mean no limit. `max_steps` stops optimization after the requested number of updates; it does not produce a trained benchmark model. For other fields use `--set KEY=JSON`, for example `--set lr=0.01` or `--set eta=2.0`. String values require JSON quotes.

The online training set is Gaussian noise, Gaussian blur, motion blur, brightness, contrast and JPEG compression. `eta=2`, `floor=0.2`, `lambda_prediction=1` and `lambda_feature=0.1` are explicit starting settings for the details not fixed numerically in the paper. `warmup_epochs=5` supplies clean-only training and `ramp_epochs=5` gradually enables consistency. Severity is `(level-1)/4`, with levels sampled uniformly from 1 to 5. Types are sampled uniformly and independently of labels.

`history.csv` records supervised, prediction, feature and total loss. `metrics.json` reports clean accuracy, all-corruption mCA and seen/unseen mCA. `corruptions.csv` retains every corruption–severity accuracy, allowing equal-weight aggregation over domains. A missing subset is represented by `null`. The comparison CSV retains separate seeds. Plots show accuracy versus severity.

## Implementation conventions

The encoder is a CIFAR-style ResNet-18. Its normalized feature enters the classifier as specified by the method equations. Clean probabilities and normalized features are detached only in the consistency objective; clean cross entropy still trains the network. Inference uses one ordinary forward pass. The online generators implement the six training transformations directly; their parameter scales are defined in `common.py`, and CIFAR-C remains the separate standardized evaluation source. Seen categories are the intersection of the training type names and the selected benchmark corruption names. Gaussian blur is an online training type outside the default 15-type CIFAR-C selection.

Data, weights and generated outputs are ignored by Git. Run directories store the effective `config.json`, CSV measurements and JSON metrics. Plots read the generated measurement files.
