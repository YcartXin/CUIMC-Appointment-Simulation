from __future__ import annotations

import unittest

from experiments import hypothesis_scenario_bank as bank_module


class GenerateBackgroundBankTest(unittest.TestCase):
    def test_produces_exact_count_per_horizon(self) -> None:
        bank = bank_module.generate_background_bank(
            n_per_horizon=15, seed=3, horizons=(7, 14)
        )
        counts = bank["horizon_days"].value_counts()
        self.assertEqual(counts.get(7), 15)
        self.assertEqual(counts.get(14), 15)

    def test_all_rows_satisfy_validity_constraints(self) -> None:
        bank = bank_module.generate_background_bank(
            n_per_horizon=20, seed=5, horizons=(7, 21)
        )
        self.assertTrue((bank["balk_low_1"] <= bank["balk_high_1"]).all())
        self.assertTrue((bank["balk_low_2"] <= bank["balk_high_2"]).all())
        self.assertTrue((bank["noshow_low_1"] <= bank["noshow_high_1"]).all())
        self.assertTrue((bank["noshow_low_2"] <= bank["noshow_high_2"]).all())
        self.assertTrue((bank["balk_threshold_1"] > bank["noshow_threshold_1"]).all())
        self.assertTrue((bank["balk_threshold_2"] > bank["noshow_threshold_2"]).all())
        self.assertTrue((bank["noshow_threshold_1"] < bank["horizon_days"] - 1).all())
        self.assertTrue((bank["noshow_threshold_2"] < bank["horizon_days"] - 1).all())

    def test_background_ids_are_unique(self) -> None:
        bank = bank_module.generate_background_bank(
            n_per_horizon=10, seed=1, horizons=(7, 14, 21, 28)
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

    def test_short_horizon_still_produces_valid_rows(self) -> None:
        # horizon=7 forces noshow_threshold < 6, i.e. only the value 4 in
        # the user's threshold grid; this must not starve the sampler.
        bank = bank_module.generate_background_bank(
            n_per_horizon=25, seed=2, horizons=(7,)
        )
        self.assertEqual(len(bank), 25)
        self.assertTrue((bank["noshow_threshold_1"] == 4).all())

    def test_lambda_derived_from_rho_and_class_share(self) -> None:
        bank = bank_module.generate_background_bank(
            n_per_horizon=10, seed=4, horizons=(14,)
        )
        implied_total = (bank["lambda_1"] + bank["lambda_2"]) / bank["slots_per_day"]
        self.assertTrue((implied_total - bank["rho"]).abs().max() < 1e-6)


if __name__ == "__main__":
    unittest.main()
