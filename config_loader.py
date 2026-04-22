from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from model import ThresholdRule, PatientClassParams, SimulationConfig


def _build_probability_rule(rule_config: Dict[str, Any]) -> ThresholdRule:
    """
    Build a probability rule object from a config dictionary.
    Currently supports threshold rules only.
    """
    rule_type = rule_config.get("type")

    if rule_type != "threshold":
        raise ValueError(f"Unsupported rule type: {rule_type}")

    return ThresholdRule(
        threshold=int(rule_config["threshold"]),
        low=float(rule_config["low"]),
        high=float(rule_config["high"]),
    )


def load_config(path: str | Path) -> SimulationConfig:
    """
    Load a YAML config file and convert it into a SimulationConfig object.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw_classes = raw["classes"]
    classes: Dict[int, PatientClassParams] = {}

    for class_id_str, class_cfg in raw_classes.items():
        class_id = int(class_id_str)

        classes[class_id] = PatientClassParams(
            class_id=class_id,
            lambda_per_slot=float(class_cfg["lambda_per_slot"]),
            balk_prob=_build_probability_rule(class_cfg["balk_prob"]),
            cancel_prob=float(class_cfg["cancel_prob"]),
            no_show_prob=_build_probability_rule(class_cfg["no_show_prob"]),
            value=float(class_cfg.get("value", 1.0)),
        )

    return SimulationConfig(
        slots_per_day=int(raw["slots_per_day"]),
        horizon_days=int(raw["horizon_days"]),
        burn_in_days=int(raw["burn_in_days"]),
        measure_days=int(raw["measure_days"]),
        cooldown_days=int(raw["cooldown_days"]),
        classes=classes,
        seed=None if raw.get("seed") is None else int(raw["seed"]),
    )