from pathlib import Path

from config_loader import load_config
from engine import ClinicAppointmentSimulation


def run_scenario(config_path: Path):
    config = load_config(config_path)
    sim = ClinicAppointmentSimulation(config)
    return sim.run()


def main() -> None:
    repo_dir = Path(__file__).resolve().parent

    baseline_path = repo_dir / "configs" / "baseline.yaml"
    scenario_2_path = repo_dir / "configs" / "scenario_2.yaml"

    baseline_results = run_scenario(baseline_path)
    scenario_2_results = run_scenario(scenario_2_path)

    print("=== Overall Comparison ===")
    print(f"Baseline scheduled utilization:    {baseline_results.overall_scheduled_utilization:.3f}")
    print(f"Scenario 2 scheduled utilization:  {scenario_2_results.overall_scheduled_utilization:.3f}")
    print()

    print(f"Baseline attended utilization:     {baseline_results.overall_attended_utilization:.3f}")
    print(f"Scenario 2 attended utilization:   {scenario_2_results.overall_attended_utilization:.3f}")
    print()

    print(f"Baseline overall percent served:   {baseline_results.overall_percent_serviced:.3f}")
    print(f"Scenario 2 overall percent served: {scenario_2_results.overall_percent_serviced:.3f}")
    print()

    print(f"Baseline total served:             {baseline_results.total_served}")
    print(f"Scenario 2 total served:           {scenario_2_results.total_served}")
    print()

    print(f"Baseline total value:              {baseline_results.total_value:.3f}")
    print(f"Scenario 2 total value:            {scenario_2_results.total_value:.3f}")

    print("\n=== Class-Level Comparison ===")
    for class_id in baseline_results.class_metrics:
        b = baseline_results.class_metrics[class_id]
        s = scenario_2_results.class_metrics[class_id]

        print(f"\nClass {class_id}")
        print(f"  Baseline arrivals:               {b.arrivals}")
        print(f"  Scenario 2 arrivals:             {s.arrivals}")

        print(f"  Baseline booked:                 {b.booked}")
        print(f"  Scenario 2 booked:               {s.booked}")

        print(f"  Baseline balked:                 {b.balked}")
        print(f"  Scenario 2 balked:               {s.balked}")

        print(f"  Baseline no offer:               {b.no_offer}")
        print(f"  Scenario 2 no offer:             {s.no_offer}")

        print(f"  Baseline canceled:               {b.canceled}")
        print(f"  Scenario 2 canceled:             {s.canceled}")

        print(f"  Baseline no-show:                {b.no_show}")
        print(f"  Scenario 2 no-show:              {s.no_show}")

        print(f"  Baseline served:                 {b.served}")
        print(f"  Scenario 2 served:               {s.served}")

        print(f"  Baseline mean delay:             {b.mean_booking_delay:.3f}")
        print(f"  Scenario 2 mean delay:           {s.mean_booking_delay:.3f}")

        print(f"  Baseline attended utilization:   {b.attended_utilization(baseline_results.total_slots):.3f}")
        print(f"  Scenario 2 attended utilization: {s.attended_utilization(scenario_2_results.total_slots):.3f}")

        print(f"  Baseline percent serviced:       {b.percent_serviced:.3f}")
        print(f"  Scenario 2 percent serviced:     {s.percent_serviced:.3f}")


if __name__ == "__main__":
    main()