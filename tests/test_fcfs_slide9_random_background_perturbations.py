from pathlib import Path

import pandas as pd

from experiments.fcfs_slide9_random_background_perturbations import (
    CONDITION_ORDER,
    DEMAND_VALUES,
    MIN_POST_PRE_GAP,
    PERTURBATION,
    apply_condition,
    create_design,
    generate_backgrounds,
)


def test_background_constraints() -> None:
    backgrounds = generate_backgrounds(100)
    for class_id in (1, 2):
        for behavior in ("balk", "noshow"):
            low = backgrounds[f"class_{class_id}_{behavior}_low"]
            high = backgrounds[f"class_{class_id}_{behavior}_high"]
            assert (high - low >= MIN_POST_PRE_GAP - 1e-12).all()
            assert (high > low + PERTURBATION).all()
            assert (high + PERTURBATION <= 1.0).all()
        assert (
            backgrounds[f"class_{class_id}_balk_threshold"]
            > backgrounds[f"class_{class_id}_noshow_threshold"]
        ).all()


def test_perturbation_definitions() -> None:
    background = generate_backgrounds(2).iloc[0].to_dict()
    baseline = apply_condition(background, "baseline")
    for condition in CONDITION_ORDER[1:]:
        changed = apply_condition(background, condition)
        changed_columns = {
            column
            for column in background
            if isinstance(background[column], float) and changed[column] != baseline[column]
        }
        expected = {
            "noshow_post_gap": {"class_1_noshow_high"},
            "noshow_pre_gap": {"class_1_noshow_low"},
            "noshow_common_post": {"class_1_noshow_high", "class_2_noshow_high"},
            "balking_post_gap": {"class_1_balk_high"},
            "balking_pre_gap": {"class_1_balk_low"},
            "balking_common_post": {"class_1_balk_high", "class_2_balk_high"},
        }[condition]
        assert changed_columns == expected
        for column in expected:
            assert abs(changed[column] - baseline[column] - PERTURBATION) < 1e-12


def test_design_size_and_pairing(tmp_path: Path) -> None:
    design = create_design(tmp_path, "smoke", n_backgrounds=3, seeds_per_background=2, design_seed=17)
    assert len(design) == 3 * len(DEMAND_VALUES) * 2 * len(CONDITION_ORDER)
    assert (design.groupby("pair_id")["condition"].nunique() == len(CONDITION_ORDER)).all()
    assert (design.groupby("pair_id")["seed"].nunique() == 1).all()
    assert set(design["condition"]) == set(CONDITION_ORDER)


def test_common_post_preserves_class_gap() -> None:
    background = generate_backgrounds(2).iloc[0].to_dict()
    for behavior, condition in (("noshow", "noshow_common_post"), ("balk", "balking_common_post")):
        changed = apply_condition(background, condition)
        before = background[f"class_1_{behavior}_high"] - background[f"class_2_{behavior}_high"]
        after = changed[f"class_1_{behavior}_high"] - changed[f"class_2_{behavior}_high"]
        assert abs(before - after) < 1e-12
