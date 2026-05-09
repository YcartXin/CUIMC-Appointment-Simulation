from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from simulation.config_loader import load_config
from simulation.engine import ClinicAppointmentSimulation
from analysis.metrics import aggregate_delay_metrics


def run_scenario(config_path: Path):
    config = load_config(config_path)
    sim = ClinicAppointmentSimulation(config)
    return sim.run()


def main() -> None:
    baseline_path = REPO_DIR / "configs" / "baseline.yaml"
    scenario_2_path = REPO_DIR / "configs" / "scenario_2.yaml"

    baseline_results = run_scenario(baseline_path)
    scenario_2_results = run_scenario(scenario_2_path)
    baseline_delays = aggregate_delay_metrics(baseline_results)
    scenario_2_delays = aggregate_delay_metrics(scenario_2_results)

    print("=== Overall Comparison ===")
    print(f"Baseline average utilization:        {baseline_results.average_utilization:.3f}")
    print(f"Scenario 2 average utilization:      {scenario_2_results.average_utilization:.3f}")
    print()

    print(f"Baseline overall percent serviced:   {baseline_results.overall_percent_serviced:.3f}")
    print(f"Scenario 2 overall percent serviced: {scenario_2_results.overall_percent_serviced:.3f}")
    print()

    print(f"Baseline mean accepted delay:       {baseline_delays['mean_accepted_booking_delay']:.3f}")
    print(f"Scenario 2 mean accepted delay:     {scenario_2_delays['mean_accepted_booking_delay']:.3f}")
    print()

    print(f"Baseline mean offered delay:        {baseline_delays['mean_offered_booking_delay']:.3f}")
    print(f"Scenario 2 mean offered delay:      {scenario_2_delays['mean_offered_booking_delay']:.3f}")

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
