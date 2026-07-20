from __future__ import annotations

import unittest

from experiments import hypothesis_scenario_bank as bank_module


class GenerateBackgroundBankTest(unittest.TestCase):
    def test_produces_exact_count_per_horizon(self) -> None:
        bank = bank_module.generate_background_bank(
            n_per_horizon=15, seed=3, horizons=(6, 14)
        )
        counts = bank["horizon_days"].value_counts()
        self.assertEqual(counts.get(6), 15)
        self.assertEqual(counts.get(14), 15)

    def test_all_rows_satisfy_validity_constraints(self) -> None:
        bank = bank_module.generate_background_bank(
            n_per_horizon=20, seed=5, horizons=(6, 22)
        )
        self.assertTrue((bank["balk_low_1"] <= bank["balk_high_1"]).all())
        self.assertTrue((bank["balk_low_2"] <= bank["balk_high_2"]).all())
        self.assertTrue((bank["noshow_low_1"] <= bank["noshow_high_1"]).all())
        self.assertTrue((bank["noshow_low_2"] <= bank["noshow_high_2"]).all())
        self.assertTrue((bank["balk_threshold_1"] > bank["noshow_threshold_1"]).all())
        self.assertTrue((bank["balk_threshold_2"] > bank["noshow_threshold_2"]).all())

    def test_background_ids_are_unique(self) -> None:
        bank = bank_module.generate_background_bank(
            n_per_horizon=10, seed=1, horizons=bank_module.HORIZON_VALUES
        )
        self.assertEqual(bank["background_id"].nunique(), len(bank))

    def test_does_not_force_ordering_between_class_thresholds(self) -> None:
        # Both H1 (threshold_1 < threshold_2) and its violation must be
        # representable in the same bank, since the condition itself is
        # being tested rather than assumed.
        bank = bank_module.generate_background_bank(
            n_per_horizon=40, seed=9, horizons=(14,)
        )
        gap = bank["noshow_threshold_2"] - bank["noshow_threshold_1"]
        self.assertTrue((gap > 0).any())
        self.assertTrue((gap <= 0).any())

    def test_shortest_horizon_still_produces_full_rows(self) -> None:
        # horizon=2 is the shortest value in HORIZON_VALUES. Thresholds are
        # NOT filtered or capped at bank-generation time (that capping now
        # happens downstream in hypothesis_common.build_config), so the
        # sampler must still fill the full row count here even though every
        # threshold value in the grid exceeds this horizon.
        bank = bank_module.generate_background_bank(
            n_per_horizon=25, seed=2, horizons=(2,)
        )
        self.assertEqual(len(bank), 25)
        self.assertTrue(bank["noshow_threshold_1"].isin(bank_module.NOSHOW_THRESHOLD_VALUES).all())
        self.assertTrue(bank["balk_threshold_1"].isin(bank_module.BALK_THRESHOLD_VALUES).all())

    def test_lambda_derived_from_rho_and_class_share(self) -> None:
        bank = bank_module.generate_background_bank(
            n_per_horizon=10, seed=4, horizons=(14,)
        )
        implied_total = (bank["lambda_1"] + bank["lambda_2"]) / bank["slots_per_day"]
        self.assertTrue((implied_total - bank["rho"]).abs().max() < 1e-6)


if __name__ == "__main__":
    unittest.main()
