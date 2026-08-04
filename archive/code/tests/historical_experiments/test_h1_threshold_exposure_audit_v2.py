from __future__ import annotations

import unittest

from experiments.h1_threshold_exposure_audit_v2 import (
    CLASS1_SHARE_VALUES,
    CAPACITY_VALUES,
    NATIVE_HORIZON_VALUES,
    RHO_VALUES,
    clinic_contexts,
    generate_bank,
    patient_profiles,
)


class ThresholdExposureAuditV2DesignTests(unittest.TestCase):
    def test_profile_count_and_family_count(self) -> None:
        profiles = patient_profiles()
        self.assertEqual(len(profiles), 18)
        counts: dict[str, int] = {}
        for profile in profiles:
            counts[profile.characteristic] = counts.get(profile.characteristic, 0) + 1
        self.assertEqual(
            counts,
            {
                "no_show_sensitivity": 6,
                "balking_sensitivity": 6,
                "joint_delay_sensitivity": 6,
            },
        )

    def test_demand_grid(self) -> None:
        self.assertEqual(RHO_VALUES, (1.2, 1.4, 1.7, 2.0, 3.0))

    def test_context_grid_is_fully_crossed(self) -> None:
        contexts = clinic_contexts()
        expected = (
            len(RHO_VALUES)
            * len(CLASS1_SHARE_VALUES)
            * len(CAPACITY_VALUES)
            * len(NATIVE_HORIZON_VALUES)
        )
        self.assertEqual(expected, 90)
        self.assertEqual(len(contexts), expected)
        self.assertFalse(contexts.duplicated().any())

    def test_bank_has_expected_size(self) -> None:
        bank = generate_bank()
        self.assertEqual(len(bank), 1620)
        self.assertFalse(bank["background_id"].duplicated().any())

    def test_probabilities_are_multiples_of_five_percent(self) -> None:
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

    def test_balking_threshold_exceeds_noshow_threshold(self) -> None:
        for profile in patient_profiles():
            self.assertGreater(profile.balk_1.threshold, profile.noshow_1.threshold)
            self.assertGreater(profile.balk_2.threshold, profile.noshow_2.threshold)

    def test_non_target_behavior_is_held_equal(self) -> None:
        for profile in patient_profiles():
            if profile.characteristic == "no_show_sensitivity":
                self.assertEqual(profile.balk_1, profile.balk_2)
                self.assertEqual(profile.cancel_1, profile.cancel_2)
            elif profile.characteristic == "balking_sensitivity":
                self.assertEqual(profile.noshow_1, profile.noshow_2)
                self.assertEqual(profile.cancel_1, profile.cancel_2)

    def test_key_revised_thresholds(self) -> None:
        profiles = {profile.profile_id: profile for profile in patient_profiles()}

        self.assertEqual(profiles["NS_LOW_MILD"].noshow_1.threshold, 7)
        self.assertEqual(profiles["NS_LOW_STRONG"].noshow_1.threshold, 3)
        self.assertEqual(profiles["NS_MODERATE_MILD"].noshow_1.threshold, 5)

        self.assertEqual(profiles["BK_LOW_MILD"].balk_1.threshold, 6)
        self.assertEqual(profiles["BK_LOW_STRONG"].balk_1.threshold, 4)
        self.assertEqual(profiles["BK_MODERATE_MILD"].balk_1.threshold, 5)

        self.assertEqual(profiles["JT_MODERATE_SAME"].noshow_1.threshold, 5)
        self.assertEqual(profiles["JT_MODERATE_SAME"].balk_1.threshold, 6)
        self.assertEqual(profiles["JT_MODERATE_STRONG"].noshow_1.threshold, 3)
        self.assertEqual(profiles["JT_MODERATE_STRONG"].balk_1.threshold, 4)


if __name__ == "__main__":
    unittest.main()
