from pathlib import Path

from config_loader import load_config
from engine import ClinicAppointmentSimulation


def run_scenario(config_path: Path):
    config = load_config(config_path)
    sim = ClinicAppointmentSimulation(config)
    return sim.run()


def aggregate_mean_accepted_booking_delay(results) -> float:
    booked = sum(m.booked for m in results.class_metrics.values())
    delay = sum(m.total_booking_delay for m in results.class_metrics.values())
    return delay / booked if booked > 0 else 0.0


def aggregate_mean_offered_booking_delay(results) -> float:
    offered = sum(m.offered for m in results.class_metrics.values())
    delay = sum(m.total_offered_booking_delay for m in results.class_metrics.values())
    return delay / offered if offered > 0 else 0.0


def main() -> None:
    repo_dir = Path(__file__).resolve().parent

    baseline_path = repo_dir / "configs" / "baseline.yaml"
    scenario_2_path = repo_dir / "configs" / "scenario_2.yaml"

    baseline_results = run_scenario(baseline_path)
    scenario_2_results = run_scenario(scenario_2_path)

    print("=== Overall Comparison ===")
    print(f"Baseline average utilization:        {baseline_results.average_utilization:.3f}")
    print(f"Scenario 2 average utilization:      {scenario_2_results.average_utilization:.3f}")
    print()

    print(f"Baseline overall percent serviced:   {baseline_results.overall_percent_serviced:.3f}")
    print(f"Scenario 2 overall percent serviced: {scenario_2_results.overall_percent_serviced:.3f}")
    print()

    print(f"Baseline mean accepted delay:       {aggregate_mean_accepted_booking_delay(baseline_results):.3f}")
    print(f"Scenario 2 mean accepted delay:     {aggregate_mean_accepted_booking_delay(scenario_2_results):.3f}")
    print()

    print(f"Baseline mean offered delay:        {aggregate_mean_offered_booking_delay(baseline_results):.3f}")
    print(f"Scenario 2 mean offered delay:      {aggregate_mean_offered_booking_delay(scenario_2_results):.3f}")

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

        print(f"  Baseline mean accepted delay:    {b.mean_accepted_booking_delay:.3f}")
        print(f"  Scenario 2 mean accepted delay:  {s.mean_accepted_booking_delay:.3f}")

        print(f"  Baseline mean offered delay:     {b.mean_offered_booking_delay:.3f}")
        print(f"  Scenario 2 mean offered delay:   {s.mean_offered_booking_delay:.3f}")

        print(f"  Baseline percent serviced:       {b.percent_serviced:.3f}")
        print(f"  Scenario 2 percent serviced:     {s.percent_serviced:.3f}")


if __name__ == "__main__":
    main()
