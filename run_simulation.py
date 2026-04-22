from pathlib import Path
from config_loader import load_config
from engine import ClinicAppointmentSimulation

def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    config = load_config(repo_dir / "configs" / "baseline.yaml")
    sim = ClinicAppointmentSimulation(config)
    results = sim.run()

    print("=== Per-class metrics ===")
    for class_id, m in results.class_metrics.items():
        print(f"\nClass {class_id}")
        print(f"  Arrivals:              {m.arrivals}")
        print(f"  Booked:                {m.booked}")
        print(f"  Balked:                {m.balked}")
        print(f"  NoOffer:               {m.no_offer}")
        print(f"  Canceled:              {m.canceled}")
        print(f"  NoShow:                {m.no_show}")
        print(f"  Served:                {m.served}")
        print(f"  Mean booking delay:    {m.mean_booking_delay:.3f}")
        print(f"  Attended utilization:  {m.attended_utilization(results.total_slots):.3f}")
        print(f"  Percent serviced:      {m.percent_serviced:.3f}")

    print("\n=== Slot metrics ===")
    sm = results.slot_metrics
    print(f"BookedSlots:             {sm.booked_slots}")
    print(f"ServedSlots:             {sm.served_slots}")
    print(f"NoShowSlots:             {sm.no_show_slots}")
    print(f"EmptySlots:              {sm.empty_slots}")

    print("\n=== Aggregate outputs ===")
    print(f"Scheduled utilization:   {results.overall_scheduled_utilization:.3f}")
    print(f"Attended utilization:    {results.overall_attended_utilization:.3f}")
    print(f"Overall percent served:  {results.overall_percent_serviced:.3f}")
    print(f"Total served:            {results.total_served}")
    print(f"Total value:             {results.total_value:.3f}")

    print("\n=== Start-of-day summary state for Day 0 ===")
    print(results.daily_summary_states[0])

    print("\n=== Final rolling full state Y_t(r,m) ===")
    for r, row in enumerate(results.final_full_state):
        print(f"r={r}: {row}")


if __name__ == "__main__":
    main()