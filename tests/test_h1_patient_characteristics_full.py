from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_full_bank_shape_and_crossing() -> None:
    bank_module = _load(
        REPO / "experiments" / "h1_patient_characteristics_full_bank.py",
        "h1_full_bank",
    )
    bank = bank_module.generate_bank()
    assert len(bank) == 3780
    assert bank["profile_id"].nunique() == 21
    assert bank["clinic_context_id"].nunique() == 180
    assert set(bank["rho"]) == {1.2, 1.4, 1.7, 2.0, 2.5, 3.0}
    assert set(bank["horizon_days"]) == {10, 14, 22}
    assert set(bank["slots_per_day"]) == {30, 50}
    counts = bank.groupby("profile_id").size()
    assert counts.eq(180).all()
    assert not bank["cap_thresholds_to_horizon"].any()


def test_profile_probabilities_and_threshold_order() -> None:
    bank_module = _load(
        REPO / "experiments" / "h1_patient_characteristics_full_bank.py",
        "h1_full_bank_profiles",
    )
    bank = bank_module.generate_bank()
    for suffix in ("1", "2"):
        assert (bank[f"balk_threshold_{suffix}"] > bank[f"noshow_threshold_{suffix}"]).all()
        assert bank[f"balk_high_{suffix}"].max() <= 0.25
        assert bank[f"noshow_high_{suffix}"].max() <= 0.25


def test_moderate_references_remain_less_sensitive_than_class1_treatments() -> None:
    bank_module = _load(
        REPO / "experiments" / "h1_patient_characteristics_full_bank.py",
        "h1_full_bank_references",
    )
    profiles = pd.DataFrame(
        [bank_module._profile_record(profile) | {
            "characteristic": profile.characteristic,
            "reference": profile.class2_reference,
            "contrast": profile.contrast_level,
        } for profile in bank_module.patient_profiles()]
    )
    no_show = profiles[
        (profiles["characteristic"] == "no_show_sensitivity")
        & (profiles["reference"] == "moderate")
        & profiles["contrast"].isin(["mild", "strong"])
    ]
    assert (no_show["noshow_threshold_2"] > no_show["noshow_threshold_1"]).all()
    balk = profiles[
        (profiles["characteristic"] == "balking_sensitivity")
        & (profiles["reference"] == "moderate")
        & profiles["contrast"].isin(["mild", "strong"])
    ]
    assert (balk["balk_threshold_2"] > balk["balk_threshold_1"]).all()


def test_supported_neutral_requires_equivalence_interval() -> None:
    post = _load(
        REPO / "analysis" / "h1_patient_characteristics_full_postprocess.py",
        "h1_full_post",
    )
    assert post._effect_status(0.001, -0.003, 0.004, 0.005) == "supported_neutral"
    assert post._effect_status(0.001, -0.010, 0.012, 0.005) == "uncertain"
    assert post._effect_status(0.006, 0.001, 0.011, 0.005) == "meaningful_gain"
    assert post._effect_status(-0.006, -0.011, -0.001, 0.005) == "meaningful_harm"
