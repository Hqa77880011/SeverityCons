import torch
from torch.nn import functional as F

from .common import corrupt, js_divergence

VARIANTS = [
    "severitycons",
    "clean",
    "fixed",
    "increasing",
    "decreasing",
    "no_prediction",
    "no_feature",
    "no_anchor",
]


def severity_weights(levels, variant, eta=2.0, floor=0.2):
    s = (levels.float() - 1) / 4
    if variant == "fixed":
        return torch.ones_like(s)
    if variant == "increasing":
        return torch.exp(eta * s)
    weights = torch.exp(-eta * s)
    return weights if variant == "decreasing" else weights.clamp_min(floor)


def objective(model, x, y, config, ramp):
    variant = config["variant"]
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    logits, features = model(x)
    supervised = F.cross_entropy(logits, y)
    if variant == "clean" or ramp == 0:
        return supervised, {
            "supervised": supervised.item(),
            "prediction": 0.0,
            "feature": 0.0,
        }
    levels = torch.randint(1, 6, (len(x),), device=x.device)
    corrupted = torch.cat(
        [
            corrupt(
                sample[None],
                config["train_corruptions"][
                    int(torch.randint(len(config["train_corruptions"]), (1,)))
                ],
                level,
            )
            for sample, level in zip(x, levels.cpu().tolist())
        ]
    )
    altered_logits, altered_features = model(corrupted)
    anchor_p = logits.softmax(1)
    anchor_z = F.normalize(features, dim=1)
    if variant != "no_anchor":
        anchor_p = anchor_p.detach()
        anchor_z = anchor_z.detach()
    weights = severity_weights(levels, variant, config["eta"], config["floor"])
    prediction = (weights * js_divergence(anchor_p, altered_logits.softmax(1))).mean()
    feature = (
        weights * (1 - (anchor_z * F.normalize(altered_features, dim=1)).sum(1))
    ).mean()
    lambda_p = 0 if variant == "no_prediction" else config["lambda_prediction"]
    lambda_f = 0 if variant == "no_feature" else config["lambda_feature"]
    loss = supervised + ramp * (lambda_p * prediction + lambda_f * feature)
    return loss, {
        "supervised": supervised.item(),
        "prediction": prediction.item(),
        "feature": feature.item(),
    }
