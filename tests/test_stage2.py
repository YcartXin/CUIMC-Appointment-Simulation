from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.robustness.stage2 import (
    HYPOTHESES,
    _parse_hypotheses,
    _select_h1,
    _select_h4,
    _select_h8,
    _select_h9,
)
from experiments.robustness.scenario_space import STAGE1_SEEDS, STAGE2_SEEDS


def _write_effects(root: Path, hypothesis: str, frame: pd.DataFrame) -> None:
    path = root / hypothesis / "summary" / f"{hypothesis}_scenario_effects.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


class TestStage2Selection(unittest.TestCase):
    def test_stage2_seeds_are_new_and_have_100_values(self):
        self.assertEqual(len(STAGE2_SEEDS), 100)
        self.assertTrue(set(STAGE1_SEEDS).isdisjoint(STAGE2_SEEDS))

    def test_parse_hypotheses(self):
        self.assertEqual(_parse_hypotheses("all"), HYPOTHESES)
        self.assertEqual(_parse_hypotheses("1,h3,7"), ("h1", "h3", "h7"))

    def test_h1_selects_reversal_and_uncertainty_limited_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            effects = pd.DataFrame(
                [
                    {
                        "background_id": "A",
                        "classification": "reversed",
                        "demand_regime": "high",
                        "delta_class1_served_rate": 0.01,
                        "delta_mean_offered_delay": 0.5,
                        "class1_served_component": "reversed",
                        "offered_delay_component": "reversed",
                    },
                    {
                        "background_id": "B",
                        "classification": "inconclusive",
                        "demand_regime": "high",
                        "delta_class1_served_rate": -0.006,
                        "delta_mean_offered_delay": -0.30,
                        "class1_served_component": "inconclusive",
                        "offered_delay_component": "inconclusive",
                    },
                    {
                        "background_id": "C",
                        "classification": "inconclusive",
                        "demand_regime": "high",
                        "delta_class1_served_rate": -0.001,
                        "delta_mean_offered_delay": -0.10,
                        "class1_served_component": "inconclusive",
                        "offered_delay_component": "inconclusive",
                    },
                ]
            )
            _write_effects(root, "h1", effects)
            selected = _select_h1(root)
            self.assertEqual(set(selected["background_id"]), {"A", "B"})

    def test_h4_omits_stably_decreasing_inconclusive_curve(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            effects = pd.DataFrame(
                [
                    {
                        "background_id": "A",
                        "classification": "inconclusive",
                        "demand_regime": "high",
                        "peak_level": 0.3,
                        "hump_rise_from_low": 0.30,
                        "hump_fall_to_high": 0.40,
                    },
                    {
                        "background_id": "B",
                        "classification": "inconclusive",
                        "demand_regime": "high",
                        "peak_level": 0.0,
                        "hump_rise_from_low": float("nan"),
                        "hump_fall_to_high": float("nan"),
                    },
                ]
            )
            _write_effects(root, "h4", effects)
            selected = _select_h4(root)
            self.assertEqual(selected["background_id"].tolist(), ["A"])

    def test_h8_selects_reversals_and_positive_uncertain_difference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            effects = pd.DataFrame(
                [
                    {
                        "background_id": "A",
                        "classification": "reversed",
                        "absolute_effect_difference": -0.01,
                    },
                    {
                        "background_id": "B",
                        "classification": "inconclusive",
                        "absolute_effect_difference": 0.003,
                    },
                    {
                        "background_id": "C",
                        "classification": "inconclusive",
                        "absolute_effect_difference": 0.001,
                    },
                ]
            )
            _write_effects(root, "h8", effects)
            selected = _select_h8(root)
            self.assertEqual(set(selected["background_id"]), {"A", "B"})

    def test_h9_requires_both_point_estimates_for_uncertain_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            effects = pd.DataFrame(
                [
                    {
                        "background_id": "A",
                        "classification": "inconclusive",
                        "common_minus_gap_utilization_effect": 0.004,
                        "gap_minus_common_served_gap_effect": 0.004,
                    },
                    {
                        "background_id": "B",
                        "classification": "inconclusive",
                        "common_minus_gap_utilization_effect": 0.004,
                        "gap_minus_common_served_gap_effect": 0.001,
                    },
                ]
            )
            _write_effects(root, "h9", effects)
            selected = _select_h9(root)
            self.assertEqual(selected["background_id"].tolist(), ["A"])


if __name__ == "__main__":
    unittest.main()
