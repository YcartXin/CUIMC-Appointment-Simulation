from __future__ import annotations

import unittest

from experiments import h1_short_horizon_reservation as h1
from experiments.h1_patient_characteristics_confirmatory_bank import (
    CAPACITY_VALUES,
    CLASS1_SHARE_VALUES,
    NATIVE_HORIZON_VALUES,
    RHO_VALUES,
    clinic_contexts,
    generate_full_bank,
    patient_profiles,
)
from experiments.hypothesis_common import (
    WEIGHTED_UTILIZATION_W1,
    WEIGHTED_UTILIZATION_W2,
)


class FullPatientCharacteristicsDesignTests(unittest.TestCase):
    def test_profile_and_background_counts(self) -> None:
        profiles = patient_profiles()
        bank = generate_full_bank()

        self.assertEqual(len(profiles), 21)
        self.assertEqual(bank["profile_id"].nunique(), 21)
        self.assertEqual(bank["clinic_context_id"].nunique(), 150)
        self.assertEqual(len(bank), 3150)
        self.assertFalse(bank["background_id"].duplicated().any())

    def test_new_demand_grid(self) -> None:
        self.assertEqual(RHO_VALUES, (1.2, 1.4, 1.7, 2.0, 3.0))

    def test_native_horizon_grid(self) -> None:
        self.assertEqual(NATIVE_HORIZON_VALUES, (10, 14, 22))

    def test_clinic_grid_is_fully_crossed(self) -> None:
        contexts = clinic_contexts()
        expected = (
            len(RHO_VALUES)
            * len(CLASS1_SHARE_VALUES)
            * len(CAPACITY_VALUES)
            * len(NATIVE_HORIZON_VALUES)
        )
        self.assertEqual(expected, 150)
        self.assertEqual(len(contexts), expected)
        self.assertFalse(
            contexts[
                [
                    "rho",
                    "class1_share",
                    "slots_per_day",
                    "horizon_days",
                ]
            ].duplicated().any()
        )

    def test_final_no_show_profiles(self) -> None:
        profiles = {profile.profile_id: profile for profile in patient_profiles()}

        self.assertEqual(
            profiles["NS_LOW_SAME"].noshow_2,
            profiles["NS_LOW_SAME"].noshow_1,
        )
        self.assertEqual(profiles["NS_LOW_SAME"].noshow_2.threshold, 14)
        self.assertEqual(profiles["NS_LOW_MILD"].noshow_1.threshold, 5)
        self.assertEqual(profiles["NS_LOW_STRONG"].noshow_1.threshold, 3)

        self.assertEqual(profiles["NS_MODERATE_SAME"].noshow_2.threshold, 8)
        self.assertEqual(profiles["NS_MODERATE_MILD"].noshow_1.threshold, 5)
        self.assertEqual(profiles["NS_MODERATE_STRONG"].noshow_1.threshold, 3)

    def test_final_balking_profiles(self) -> None:
        profiles = {profile.profile_id: profile for profile in patient_profiles()}

        self.assertEqual(profiles["BK_LOW_SAME"].balk_2.threshold, 16)
        self.assertEqual(profiles["BK_LOW_MILD"].balk_1.threshold, 6)
        self.assertEqual(profiles["BK_LOW_STRONG"].balk_1.threshold, 4)

        self.assertEqual(profiles["BK_MODERATE_SAME"].balk_2.threshold, 9)
        self.assertEqual(profiles["BK_MODERATE_MILD"].balk_1.threshold, 6)
        self.assertEqual(profiles["BK_MODERATE_STRONG"].balk_1.threshold, 4)

    def test_moderate_reference_sits_above_class1_contrasts(self) -> None:
        profiles = {profile.profile_id: profile for profile in patient_profiles()}

        self.assertGreater(
            profiles["NS_MODERATE_MILD"].noshow_2.threshold,
            profiles["NS_MODERATE_MILD"].noshow_1.threshold,
        )
        self.assertGreater(
            profiles["NS_MODERATE_STRONG"].noshow_2.threshold,
            profiles["NS_MODERATE_STRONG"].noshow_1.threshold,
        )
        self.assertGreater(
            profiles["BK_MODERATE_MILD"].balk_2.threshold,
            profiles["BK_MODERATE_MILD"].balk_1.threshold,
        )
        self.assertGreater(
            profiles["BK_MODERATE_STRONG"].balk_2.threshold,
            profiles["BK_MODERATE_STRONG"].balk_1.threshold,
        )

    def test_cancellation_profiles_are_unchanged(self) -> None:
        profiles = {
            profile.profile_id: profile
            for profile in patient_profiles()
            if profile.characteristic == "cancellation_propensity"
        }
        self.assertEqual(profiles["CN_LOW_SAME"].cancel_1, 0.10)
        self.assertEqual(profiles["CN_LOW_MILD"].cancel_1, 0.20)
        self.assertEqual(profiles["CN_LOW_STRONG"].cancel_1, 0.30)
        for profile in profiles.values():
            self.assertEqual(profile.cancel_2, 0.10)
            self.assertEqual(profile.balk_1, profile.balk_2)
            self.assertEqual(profile.noshow_1, profile.noshow_2)

    def test_non_target_behaviour_is_held_equal(self) -> None:
        for profile in patient_profiles():
            if profile.characteristic == "no_show_sensitivity":
                self.assertEqual(profile.balk_1, profile.balk_2)
                self.assertEqual(profile.cancel_1, profile.cancel_2)
            elif profile.characteristic == "balking_sensitivity":
                self.assertEqual(profile.noshow_1, profile.noshow_2)
                self.assertEqual(profile.cancel_1, profile.cancel_2)

    def test_delay_probabilities_are_multiples_of_five_and_bounded(self) -> None:
        for profile in patient_profiles():
            for rule in (
                profile.balk_1,
                profile.balk_2,
                profile.noshow_1,
                profile.noshow_2,
            ):
                for probability in (rule.low, rule.high):
                    self.assertAlmostEqual((probability * 100) % 5, 0.0)
                    self.assertLessEqual(probability, 0.25)

    def test_thresholds_are_preserved(self) -> None:
        bank = generate_full_bank()
        self.assertTrue((bank["cap_thresholds_to_horizon"] == False).all())

    def test_both_objectives_and_weights_are_configured(self) -> None:
        self.assertIn("average_utilization", h1.OPTIMIZATION_OBJECTIVES)
        self.assertIn("weighted_utilization", h1.OPTIMIZATION_OBJECTIVES)
        self.assertEqual(WEIGHTED_UTILIZATION_W1, 2.0)
        self.assertEqual(WEIGHTED_UTILIZATION_W2, 1.0)


if __name__ == "__main__":
    unittest.main()
