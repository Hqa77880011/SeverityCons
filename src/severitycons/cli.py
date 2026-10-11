from pathlib import Path
import json
import numpy as np
import torch

from .common import (
    Images,
    CorruptionImages,
    FeatureNet,
    loader,
    accuracy,
    device_of,
    seed_everything,
    prepare,
    parse_config,
    save_json,
    save_rows,
    plot_csv,
)
from .method import objective, VARIANTS


def train(config):
    seed_everything(config["seed"])
    dataset = Images(config, "train", augment=True)
    model = FeatureNet(len(dataset.classes), normalized_head=True).to(device_of(config))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=0.9,
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    history = []
    step = 0
    for epoch in range(config["epochs"]):
        model.train()
        ramp = min(
            1.0,
            max(
                0.0,
                (epoch - config["warmup_epochs"] + 1) / max(1, config["ramp_epochs"]),
            ),
        )
        sums = {}
        samples = 0
        for x, y in loader(dataset, config, shuffle=True):
            x, y = x.to(device_of(config)), y.to(device_of(config))
            optimizer.zero_grad(set_to_none=True)
            loss, values = objective(model, x, y, config, ramp)
            loss.backward()
            optimizer.step()
            for key, value in dict(values, loss=loss.item()).items():
                sums[key] = sums.get(key, 0.0) + value * len(y)
            samples += len(y)
            step += 1
            if config["max_steps"] and step >= config["max_steps"]:
                break
        if not samples:
            raise ValueError("Training dataset is empty")
        row = {
            "epoch": epoch + 1,
            "variant": config["variant"],
            "seed": config["seed"],
            "ramp": ramp,
            **{key: value / samples for key, value in sums.items()},
            "val_accuracy": accuracy(model, Images(config, "val"), config),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        scheduler.step()
        if config["max_steps"] and step >= config["max_steps"]:
            break
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "classes": dataset.classes,
            "config": config,
            "steps": step,
        },
        output / "model.pt",
    )
    save_rows(output / "history.csv", history)
    save_json(output / "config.json", config)
    return str(output / "model.pt")


def evaluate(config):
    state = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    model = FeatureNet(len(state["classes"]), normalized_head=True).to(
        device_of(config)
    )
    model.load_state_dict(state["model"])
    clean = accuracy(model, Images(config, "test"), config)
    rows = []
    if config.get("corruption_root"):
        for name in config["eval_corruptions"]:
            for severity in config["severities"]:
                domain = CorruptionImages(
                    config["corruption_root"], name, severity, config["limit"]
                )
                rows.append(
                    {
                        "variant": state["config"]["variant"],
                        "seed": state["config"]["seed"],
                        "corruption": name,
                        "severity": severity,
                        "seen": name in config["train_corruptions"],
                        "accuracy": accuracy(model, domain, config),
                    }
                )
    summary = {
        "variant": state["config"]["variant"],
        "seed": state["config"]["seed"],
        "clean_accuracy": clean,
        "mca": float(np.mean([r["accuracy"] for r in rows])) if rows else None,
    }
    for seen, label in [(True, "seen_mca"), (False, "unseen_mca")]:
        values = [r["accuracy"] for r in rows if r["seen"] == seen]
        summary[label] = float(np.mean(values)) if values else None
    save_json(Path(config["output"]) / "metrics.json", summary)
    if rows:
        save_rows(Path(config["output"]) / "corruptions.csv", rows)
    print(json.dumps(summary), flush=True)
    return summary, rows


def compare(config):
    summaries, domains = [], []
    for seed in config["seeds"]:
        for variant in config["variants"]:
            if variant not in VARIANTS:
                raise ValueError(f"Unknown variant: {variant}")
            current = dict(
                config,
                seed=seed,
                variant=variant,
                output=str(Path(config["output"]) / variant / str(seed)),
            )
            current["checkpoint"] = train(current)
            summary, rows = evaluate(current)
            summaries.append(summary)
            domains.extend(rows)
    save_rows(Path(config["output"]) / "comparison.csv", summaries)
    if domains:
        save_rows(Path(config["output"]) / "corruptions.csv", domains)
    return summaries


def main():
    command, config = parse_config(["prepare", "train", "evaluate", "compare", "plot"])
    if command == "prepare":
        prepare(config)
    elif command == "train":
        train(config)
    elif command == "evaluate":
        evaluate(config)
    elif command == "compare":
        compare(config)
    else:
        root = Path(config["output"])
        if (root / "corruptions.csv").exists():
            plot_csv(
                root / "corruptions.csv",
                root / "accuracy_by_severity.pdf",
                "severity",
                "accuracy",
            )
        else:
            plot_csv(
                root / "history.csv", root / "training.pdf", "epoch", "val_accuracy"
            )
