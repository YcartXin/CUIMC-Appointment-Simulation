from pathlib import Path

import pandas as pd

from experiments.fcfs_slide9_behavior_comparison import (
    CANCELLATION_PROBABILITY,
    DEMAND_VALUES,
    PRE_THRESHOLD_PROBABILITY,
    PROFILE_ORDER,
    PROFILES,
    SEVERITY_VALUES,
    THRESHOLD_DAYS,
    config_for_design_row,
    create_design,
)
from simulation.config_loader import load_config
from experiments.fcfs_slide9_behavior_comparison import CONFIG_PATH


def test_design_sizes_and_shared_low_profile(tmp_path: Path) -> None:
    audit = create_design(tmp_path / "audit", "audit", 10)
    full = create_design(tmp_path / "full", "full", 100)
    assert len(audit) == len(DEMAND_VALUES) * 10
    assert len(full) == len(PROFILE_ORDER) * len(DEMAND_VALUES) * 100
    assert set(audit["profile_id"]) == {"low_low"}
    assert list(full["profile_id"].drop_duplicates()) == PROFILE_ORDER


def test_configuration_holds_all_controls() -> None:
    base = load_config(CONFIG_PATH)
    for profile_id, profile in PROFILES.items():
        row = pd.Series(
            {
                "seed": 7,
                "total_arrivals_per_day": 110.0,
                **profile,
            }
        )
        config = config_for_design_row(base, row)
        assert config.seed == 7
        for class_id in (1, 2):
            cls = config.classes[class_id]
            assert cls.lambda_per_day == 55.0
            assert cls.cancel_prob == CANCELLATION_PROBABILITY
            assert cls.balk_prob.threshold == THRESHOLD_DAYS
            assert cls.no_show_prob.threshold == THRESHOLD_DAYS
            assert cls.balk_prob.low == PRE_THRESHOLD_PROBABILITY
            assert cls.no_show_prob.low == PRE_THRESHOLD_PROBABILITY
        assert config.classes[1].balk_prob.high == profile["class_1_balk_high"]
        assert config.classes[2].balk_prob.high == SEVERITY_VALUES["low"]
        assert config.classes[1].no_show_prob.high == profile["class_1_noshow_high"]
        assert config.classes[2].no_show_prob.high == SEVERITY_VALUES["low"]


def test_design_refuses_to_mix_existing_run(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    create_design(output, "audit", 2)
    raw = output / "raw"
    raw.mkdir()
    (raw / "shard_0000_of_0001.csv").write_text("task_id\n0\n", encoding="utf-8")
    try:
        create_design(output, "audit", 2)
    except RuntimeError as exc:
        assert "Existing run output" in str(exc)
    else:
        raise AssertionError("Expected stale-output safeguard to reject the design")
